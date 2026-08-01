"""Canonical public model names for MFQ artifacts."""

from __future__ import annotations


UD_RECIPE_TO_MFQ_TIER = {
    "IQ2_M": "V2-M",
    "IQ2_XXS": "V2-XXS",
    "IQ3_S": "V3-S",
    "IQ3_XXS": "V3-XXS",
    "IQ4_NL": "V4-NL",
    "IQ4_XS": "V4-XS",
    "Q2_K_XL": "S2-L",
    "Q3_K_M": "S3-M",
    "Q3_K_XL": "S3-L",
    "Q4_K_M": "S4-M",
    "Q4_K_S": "S4-S",
    "Q4_K_XL": "S4-L",
    "Q5_K_M": "S5-M",
    "Q5_K_S": "S5-S",
    "Q5_K_XL": "S5-L",
    "Q6_K": "S6",
    "Q6_K_XL": "S6-L",
    "Q8_K_XL": "S8-L",
}


def mfq_tier_for_ud_recipe(recipe_name: str) -> str:
    """Return the registered MFQ public tier for a UD recipe."""

    try:
        return UD_RECIPE_TO_MFQ_TIER[recipe_name]
    except KeyError as exc:
        raise ValueError(
            f"UD recipe has no registered MFQ public tier: {recipe_name}"
        ) from exc


def canonical_mfq_filename(base_model: str, recipe_name: str) -> str:
    """Build ``<base>-MFQ-<tier>.mfq`` for a registered recipe."""

    if not base_model or "/" in base_model or "\\" in base_model:
        raise ValueError(f"invalid base model name: {base_model!r}")
    return f"{base_model}-MFQ-{mfq_tier_for_ud_recipe(recipe_name)}.mfq"
