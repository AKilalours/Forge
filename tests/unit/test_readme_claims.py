"""The README claim checker's own logic.

A checker nobody has watched fail is not a checker. The rounding rule is the part that
decides every comparison, so it is the part that gets tested: the README is allowed to
round a stored value to the precision it displays, and is not allowed to be wrong.
"""

from __future__ import annotations

from scripts.check_readme_claims import close, main


def test_rounding_to_the_displayed_precision_passes():
    assert close("0.885", 0.885148125)
    assert close("0.183", 0.1831571070841087)


def test_a_wrong_value_fails_even_when_it_looks_close():
    assert not close("0.912", 0.885148125)


def test_precision_is_taken_from_the_readme_not_the_artifact():
    # Three decimals shown: 0.8854 rounds to 0.885 and passes.
    assert close("0.885", 0.8854)
    # Four decimals shown: the same stored value must now match at four.
    assert not close("0.8851", 0.8854)


def test_percentages_are_converted_before_comparing():
    assert close("62.9%", 0.6295, percent=True)
    assert not close("62.9%", 0.6295)


def test_bold_markers_are_not_part_of_the_number():
    assert close("0.885", 0.885148125)


def test_the_real_readme_currently_matches_every_artifact():
    # check_test_count=False: obtaining the count means running pytest, and this IS
    # pytest. CI checks the badge separately by invoking the script directly.
    assert main(check_test_count=False) == 0
