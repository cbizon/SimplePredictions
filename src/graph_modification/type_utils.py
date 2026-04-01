#!/usr/bin/env python3
"""Utilities for working with biolink types.

Prefer the Biolink Model Toolkit when it can be initialized locally. If the
toolkit is unavailable or tries to fetch schema data over the network, fall
back to a small local specificity map so graph generation and tests still work
offline.
"""

from typing import Optional

try:
    import bmt
except ImportError:  # pragma: no cover - exercised when bmt is not installed
    bmt = None


_FALLBACK_ANCESTOR_COUNTS = {
    "NamedThing": 0,
    "PhysicalEssenceOrOccurrent": 1,
    "Occurrent": 1,
    "PhysicalEssence": 2,
    "BiologicalEntity": 2,
    "ChemicalOrDrugOrTreatment": 2,
    "ChemicalEntityOrGeneOrGeneProduct": 2,
    "ChemicalEntityOrProteinOrPolypeptide": 2,
    "MolecularEntity": 3,
    "DiseaseOrPhenotypicFeature": 3,
    "GeneOrGeneProduct": 3,
    "ChemicalEntity": 4,
    "BiologicalProcess": 4,
    "OrganismTaxon": 4,
    "Disease": 5,
    "Drug": 5,
    "Gene": 5,
    "Pathway": 5,
    "Protein": 5,
    "SmallMolecule": 6,
}

# Global cache for ancestor counts.
_ANCESTOR_CACHE = {}
_TOOLKIT = None
_TOOLKIT_UNAVAILABLE = False


def _get_toolkit() -> Optional["bmt.Toolkit"]:
    """Get or create a Biolink toolkit singleton."""
    global _TOOLKIT, _TOOLKIT_UNAVAILABLE
    if _TOOLKIT_UNAVAILABLE or bmt is None:
        return None
    if _TOOLKIT is None:
        try:
            _TOOLKIT = bmt.Toolkit()
        except Exception:
            _TOOLKIT_UNAVAILABLE = True
            return None
    return _TOOLKIT


def _fallback_ancestor_count(biolink_type: str) -> int:
    """Return a best-effort local specificity score for offline use."""
    if biolink_type in _FALLBACK_ANCESTOR_COUNTS:
        return _FALLBACK_ANCESTOR_COUNTS[biolink_type]

    if "Or" in biolink_type:
        return 1
    if biolink_type.endswith(("Thing", "Entity")):
        return 1
    if biolink_type.endswith(("Process", "Taxon", "Pathway")):
        return 4

    return -1


def _get_ancestor_count(biolink_type: str) -> int:
    """
    Get the number of ancestors for a biolink type (cached).

    Args:
        biolink_type: Type without biolink: prefix (e.g., "SmallMolecule")

    Returns:
        Number of ancestors
    """
    if biolink_type not in _ANCESTOR_CACHE:
        toolkit = _get_toolkit()
        if toolkit is not None:
            try:
                if toolkit.is_category(biolink_type):
                    ancestors = toolkit.get_ancestors(biolink_type)
                    _ANCESTOR_CACHE[biolink_type] = len(ancestors)
                else:
                    _ANCESTOR_CACHE[biolink_type] = -1
            except Exception:
                _ANCESTOR_CACHE[biolink_type] = _fallback_ancestor_count(biolink_type)
        else:
            _ANCESTOR_CACHE[biolink_type] = _fallback_ancestor_count(biolink_type)

    return _ANCESTOR_CACHE[biolink_type]


def get_most_specific_type(categories: list) -> str:
    """
    Given a list of biolink categories, return the most specific one.

    Most specific = fewest ancestors in the biolink hierarchy.

    Args:
        categories: List of biolink types (e.g., ["biolink:SmallMolecule", "biolink:ChemicalEntity"])

    Returns:
        Most specific type (with biolink: prefix stripped), or "Unknown" if empty

    Example:
        >>> get_most_specific_type([
        ...     "biolink:PhysicalEssenceOrOccurrent",
        ...     "biolink:BiologicalProcess"
        ... ])
        'BiologicalProcess'
    """
    if not categories:
        return "Unknown"

    # For each category, count ancestors using cache
    type_depths = []
    for category in categories:
        if not isinstance(category, str):
            continue

        # Remove biolink: prefix for bmt
        clean_type = category.replace('biolink:', '')

        # Get ancestor count (cached)
        ancestor_count = _get_ancestor_count(clean_type)

        # Skip invalid types
        if ancestor_count < 0:
            continue

        # More ancestors = more specific (further down the tree)
        # Fewer ancestors = more abstract (closer to root)
        type_depths.append((ancestor_count, clean_type))

    if not type_depths:
        # If no valid categories found, just return first one (stripped)
        first = categories[0]
        if isinstance(first, str):
            return first.replace('biolink:', '')
        return "Unknown"

    # Sort by number of ancestors (descending = most specific first)
    # More ancestors = further down the tree = more specific
    type_depths.sort(reverse=True)

    return type_depths[0][1]


if __name__ == "__main__":
    # Test with some examples
    test_cases = [
        # PhysicalEssenceOrOccurrent is abstract, BiologicalProcess is more specific
        ["biolink:PhysicalEssenceOrOccurrent", "biolink:BiologicalProcess"],

        # SmallMolecule is more specific than ChemicalEntity
        ["biolink:ChemicalEntity", "biolink:SmallMolecule"],

        # Disease is more specific than DiseaseOrPhenotypicFeature
        ["biolink:DiseaseOrPhenotypicFeature", "biolink:Disease"],
    ]

    for categories in test_cases:
        most_specific = get_most_specific_type(categories)
        print(f"{categories}")
        print(f"  → Most specific: {most_specific}\n")
