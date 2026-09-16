from forge.modeling.windowing import aggregate, windows


def test_short_document_is_one_window():
    assert windows(300) == [(0, 300)]


def test_windows_overlap_by_the_configured_amount():
    w = windows(1500, size=512, stride=384)
    assert w[0] == (0, 512)
    assert w[1] == (384, 896)
    overlap = w[0][1] - w[1][0]
    assert overlap == 128


def test_windows_cover_the_whole_document():
    n = 2000
    w = windows(n, 512, 384)
    assert w[0][0] == 0 and w[-1][1] == n
    for a, b in zip(w, w[1:], strict=False):
        assert b[0] < a[1], "a gap between windows would create a blind spot"


def test_aggregate_mean():
    assert aggregate([0.0, 1.0]) == 0.5
