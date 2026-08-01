from __future__ import annotations

import pytest

from mfq.model_naming import canonical_mfq_filename, mfq_tier_for_ud_recipe


@pytest.mark.parametrize(
    ("recipe", "tier"),
    [
        ("IQ2_M", "V2-M"),
        ("IQ2_XXS", "V2-XXS"),
        ("IQ3_S", "V3-S"),
        ("IQ3_XXS", "V3-XXS"),
        ("IQ4_NL", "V4-NL"),
        ("IQ4_XS", "V4-XS"),
        ("Q2_K_XL", "S2-L"),
        ("Q3_K_M", "S3-M"),
        ("Q3_K_XL", "S3-L"),
        ("Q4_K_M", "S4-M"),
        ("Q4_K_S", "S4-S"),
        ("Q4_K_XL", "S4-L"),
        ("Q5_K_M", "S5-M"),
        ("Q5_K_S", "S5-S"),
        ("Q5_K_XL", "S5-L"),
        ("Q6_K", "S6"),
        ("Q6_K_XL", "S6-L"),
        ("Q8_K_XL", "S8-L"),
    ],
)
def test_registered_ud_recipe_names(recipe: str, tier: str) -> None:
    assert mfq_tier_for_ud_recipe(recipe) == tier
    assert canonical_mfq_filename("Qwen3.5-9B", recipe) == (
        f"Qwen3.5-9B-MFQ-{tier}.mfq"
    )


def test_unregistered_recipe_is_rejected() -> None:
    with pytest.raises(ValueError, match="no registered MFQ public tier"):
        mfq_tier_for_ud_recipe("Q1_K")


@pytest.mark.parametrize("base", ["", "org/model", r"org\model"])
def test_invalid_base_model_is_rejected(base: str) -> None:
    with pytest.raises(ValueError, match="invalid base model name"):
        canonical_mfq_filename(base, "Q4_K_XL")
