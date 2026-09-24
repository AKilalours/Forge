"""FORGE-Base: one encoder, two heads.

    text -> tokenizer -> transformer encoder -> contextual embeddings
                                                   |              |
                                           document head      token head
                                          (human vs ai)  (human/assisted/generated)

Deliberately not a mixture of experts. Reproducing a production MoE adds engineering risk
without touching the research question, which is about which data to generate, not about
architecture. If the data thesis is right it should show up on a plain encoder.

The two heads share the encoder on purpose. Token-level supervision is a strong
regulariser for the document decision: a model forced to say WHERE the AI text is cannot
satisfy the loss with a document-level shortcut like "this reads formal, call it AI".

torch is imported lazily so the rest of the package, including the evaluation lab and all
the pure functions, works on a machine without it.
"""

from __future__ import annotations

from dataclasses import dataclass

from forge.common.schemas import TokenLabel

N_TOKEN_CLASSES = len(TokenLabel)
N_DOC_CLASSES = 2


@dataclass
class ForgeConfig:
    backbone: str = "microsoft/deberta-v3-base"
    max_length: int = 512
    stride: int = 384
    dropout: float = 0.1
    token_loss_weight: float = 0.5   # ablate this; 0.0 turns off the token head entirely
    doc_loss_weight: float = 1.0


def build_model(config: ForgeConfig, pretrained: bool = True):  # pragma: no cover - needs torch
    """Construct the model. Imports torch lazily and fails with a useful message.

    `pretrained=False` builds the architecture WITHOUT fetching the backbone's pretrained
    weights. It exists for serving, where the very next thing that happens is
    load_checkpoint overwriting every encoder tensor with a trained one. Fetching them
    first downloads 371 MB, writes it to disk and allocates a full copy of the encoder,
    all of it discarded microseconds later, on a host that has 2.7 GB in total and is
    about to load a 735 MB checkpoint.

    It cannot silently serve random weights: load_checkpoint calls load_state_dict with
    strict=True by default, so a checkpoint that does not cover every parameter raises
    instead of leaving initialised noise in the gaps.

    Training keeps the default. Fine-tuning from random init would be a different
    experiment.
    """
    try:
        import torch  # noqa: F401  # the import IS the availability probe
        from torch import nn
        from transformers import AutoConfig, AutoModel
    except ImportError as e:
        raise RuntimeError(
            "Training needs the `train` extra. Run: pip install -e '.[train]'"
        ) from e

    class ForgeBase(nn.Module):
        def __init__(self, cfg: ForgeConfig) -> None:
            super().__init__()
            self.cfg = cfg
            hf_cfg = AutoConfig.from_pretrained(cfg.backbone)
            self.encoder = (
                AutoModel.from_pretrained(cfg.backbone, config=hf_cfg)
                if pretrained
                else AutoModel.from_config(hf_cfg)
            )
            hidden = hf_cfg.hidden_size
            self.dropout = nn.Dropout(cfg.dropout)
            self.doc_head = nn.Linear(hidden, N_DOC_CLASSES)
            self.token_head = nn.Linear(hidden, N_TOKEN_CLASSES)

        def forward(self, input_ids, attention_mask, doc_labels=None, token_labels=None):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            h = self.dropout(out.last_hidden_state)          # (B, T, H)

            # Masked mean pooling, not [CLS]. DeBERTa has no NSP-pretrained [CLS], and
            # mean pooling makes the document score an average over positions, which is
            # consistent with how the token head sees the window.
            mask = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

            doc_logits = self.doc_head(pooled)               # (B, 2)
            token_logits = self.token_head(h)                # (B, T, 3)

            loss = None
            if doc_labels is not None:
                ce = nn.CrossEntropyLoss()
                loss = self.cfg.doc_loss_weight * ce(doc_logits, doc_labels)
                if token_labels is not None and self.cfg.token_loss_weight > 0:
                    # A BATCH WHERE EVERY TOKEN LABEL IS IGNORED IS NOT A ZERO LOSS, IT IS
                    # A NAN. CrossEntropyLoss reduces by mean, so with every position set
                    # to ignore_index the denominator is zero and the loss comes back NaN
                    # with no warning of any kind. The training loop then raises
                    # "non-finite loss at step 1. With DeBERTa-v3 this is usually fp16
                    # overflow", which is a canned hint that was wrong: the run was bf16,
                    # which has fp32's exponent range and does not overflow that way. An
                    # error that names a plausible wrong cause costs more than one that
                    # says only that something is wrong.
                    #
                    # Token labels come from character spans mapped through the
                    # tokenizer's offset mapping. They come back empty when the tokenizer
                    # returns no usable offsets, which is what a transformers major
                    # version bump can quietly change. So the condition is checked where
                    # it is known, and the message says which of the two it is.
                    valid = (token_labels != -100).sum()
                    if valid == 0:
                        raise RuntimeError(
                            "every token label in this batch is ignore_index. The token "
                            "head would reduce over zero elements and return NaN, which "
                            "surfaces later as a non-finite loss blamed on precision. "
                            "Token labels are built from character spans through the "
                            "tokenizer's offset mapping; check the tokenizer version and "
                            "that the spans are non-empty. Set token_loss_weight to 0.0 "
                            "to train the document head alone."
                        )
                    tce = nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss + self.cfg.token_loss_weight * tce(
                        token_logits.reshape(-1, N_TOKEN_CLASSES), token_labels.reshape(-1)
                    )
            return {"loss": loss, "doc_logits": doc_logits, "token_logits": token_logits}

    return ForgeBase(config)
