// Row gather and its scatter-add backward, fused for DeBERTa-v3's relative position table.
//
// WHY THIS OP AND NOT ATTENTION. The profile, at the shapes configs/training/baseline.yaml
// actually trains with, on an A100-80GB, says:
//
//     aten::scatter_add_                    20.87% of device time
//     aten::bmm + aten::mm combined          4.47%
//
// The matrix multiplies are not the bottleneck; the indexing around them is. DeBERTa-v3's
// disentangled attention gathers content-to-position and position-to-content scores from a
// shared relative-position bucket table, and the backward of those gathers scatters the
// gradients back into it, with ReduceAdd, 72 times per step. Writing a faster GEMM would
// have been resume decoration for a stage that is 3 percent of the step.
//
// WHAT MAKES THIS FASTER THAN aten::scatter_add_. ATen's scatter_add_ is general: it works
// for any dim, any index shape, and it computes an index offset per ELEMENT through
// TensorIterator. The access pattern here is far narrower than that generality allows:
// entire contiguous rows of width `dim` move to or from a row of the table, so the base
// offset can be computed once per row and the row itself copied with coalesced, optionally
// vectorised accesses. One block per row, threads striding along the feature dimension.
//
// DETERMINISM, STATED RATHER THAN ASSUMED. The backward accumulates with atomicAdd, so when
// two source rows target the same table row the summation order is whatever the scheduler
// produces, and float addition is not associative. Results therefore differ in the last bits
// between runs. This is NOT a regression introduced here: aten::scatter_add_ on CUDA is
// nondeterministic for exactly the same reason, which is why torch.use_deterministic_algorithms
// raises on it. The tests compare against the ATen reference with a tolerance and say so.
//
// gpuAtomicAdd rather than a raw atomicAdd: it dispatches to the right instruction per dtype
// and provides the half and bfloat16 paths, which raw atomicAdd only has on newer
// architectures. Training runs bf16, so a float-only kernel would be useless here.

#include <torch/extension.h>

#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int kThreads = 256;

template <typename scalar_t>
__global__ void gather_rows_kernel(
    const scalar_t* __restrict__ table,
    const int64_t* __restrict__ index,
    scalar_t* __restrict__ out,
    const int64_t n_rows,
    const int64_t dim,
    const int64_t table_rows) {
  const int64_t row = blockIdx.x;
  if (row >= n_rows) {
    return;
  }
  const int64_t src = index[row];
  // An out-of-range bucket index silently reads another row's embedding and trains on it.
  // CUDA_KERNEL_ASSERT turns that into a device-side failure naming this kernel.
  CUDA_KERNEL_ASSERT(src >= 0 && src < table_rows);

  const scalar_t* src_ptr = table + src * dim;
  scalar_t* dst_ptr = out + row * dim;
  for (int64_t c = threadIdx.x; c < dim; c += blockDim.x) {
    dst_ptr[c] = src_ptr[c];
  }
}

template <typename scalar_t>
__global__ void scatter_add_rows_kernel(
    const scalar_t* __restrict__ grad_out,
    const int64_t* __restrict__ index,
    scalar_t* __restrict__ grad_table,
    const int64_t n_rows,
    const int64_t dim,
    const int64_t table_rows) {
  const int64_t row = blockIdx.x;
  if (row >= n_rows) {
    return;
  }
  const int64_t dst = index[row];
  CUDA_KERNEL_ASSERT(dst >= 0 && dst < table_rows);

  const scalar_t* src_ptr = grad_out + row * dim;
  scalar_t* dst_ptr = grad_table + dst * dim;
  // The base offset is computed once here. ATen recomputes an equivalent offset for every
  // element, which is the generality this kernel trades away.
  for (int64_t c = threadIdx.x; c < dim; c += blockDim.x) {
    gpuAtomicAdd(dst_ptr + c, src_ptr[c]);
  }
}

void check_inputs(const at::Tensor& table, const at::Tensor& index) {
  TORCH_CHECK(table.is_cuda(), "table must be a CUDA tensor");
  TORCH_CHECK(index.is_cuda(), "index must be a CUDA tensor");
  TORCH_CHECK(table.dim() == 2, "table must be 2-D (rows, dim), got ", table.dim(), "-D");
  TORCH_CHECK(index.dim() == 1, "index must be 1-D, got ", index.dim(), "-D");
  TORCH_CHECK(index.scalar_type() == at::kLong, "index must be int64");
  TORCH_CHECK(table.is_contiguous(), "table must be contiguous");
  TORCH_CHECK(index.is_contiguous(), "index must be contiguous");
}

}  // namespace

at::Tensor gather_rows(const at::Tensor& table, const at::Tensor& index) {
  check_inputs(table, index);
  const at::cuda::CUDAGuard guard(table.device());

  const int64_t n_rows = index.size(0);
  const int64_t dim = table.size(1);
  auto out = at::empty({n_rows, dim}, table.options());
  if (n_rows == 0) {
    return out;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, table.scalar_type(), "gather_rows", [&] {
        gather_rows_kernel<scalar_t>
            <<<n_rows, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
                table.data_ptr<scalar_t>(),
                index.data_ptr<int64_t>(),
                out.data_ptr<scalar_t>(),
                n_rows,
                dim,
                table.size(0));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor scatter_add_rows(
    const at::Tensor& grad_out,
    const at::Tensor& index,
    const int64_t table_rows) {
  TORCH_CHECK(grad_out.is_cuda(), "grad_out must be a CUDA tensor");
  TORCH_CHECK(grad_out.dim() == 2, "grad_out must be 2-D");
  TORCH_CHECK(index.scalar_type() == at::kLong, "index must be int64");
  const at::cuda::CUDAGuard guard(grad_out.device());

  auto grad = grad_out.contiguous();
  const int64_t n_rows = grad.size(0);
  const int64_t dim = grad.size(1);
  auto grad_table = at::zeros({table_rows, dim}, grad.options());
  if (n_rows == 0) {
    return grad_table;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, grad.scalar_type(), "scatter_add_rows", [&] {
        scatter_add_rows_kernel<scalar_t>
            <<<n_rows, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad.data_ptr<scalar_t>(),
                index.contiguous().data_ptr<int64_t>(),
                grad_table.data_ptr<scalar_t>(),
                n_rows,
                dim,
                table_rows);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_table;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_rows", &gather_rows, "Gather whole rows of a 2-D table by index (CUDA)");
  m.def("scatter_add_rows", &scatter_add_rows,
        "Accumulate whole rows into a zeroed table by index (CUDA)");
}
