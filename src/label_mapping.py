"""
Label Mapping Module
====================

Converts ECG annotation symbols to standardized beat classes for machine learning.

The default mapping follows the common AAMI-style grouping:
- Normal beats
- Supraventricular ectopic beats (SVEB)
- Ventricular ectopic beats (PVC)
- Other / unclassifiable beats

A binary mapping is also available for the initial Normal-vs-Abnormal classifier.

Research/educational use only.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

NORMAL_SYMBOLS = {"N", "L", "R", "e", "j"}
PVC_SYMBOLS = {"V", "W", "F"}
SVEB_SYMBOLS = {"A", "a", "J", "S", "s"}
OTHER_SYMBOLS = {
    "Q", "P", "p", "T", "/", "!", "[", "]", "x", "+", "f",
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
}


NON_BEAT_SYMBOLS = {"+", "~", "|", "!", "[", "]"}


def is_beat_annotation(symbol: str) -> bool:
    """Return True if annotation represents an actual cardiac beat rather than rhythm/comment."""
    if not symbol or symbol in NON_BEAT_SYMBOLS:
        return False
    return symbol in (NORMAL_SYMBOLS | PVC_SYMBOLS | SVEB_SYMBOLS | OTHER_SYMBOLS)


def map_symbol_to_class(symbol: str, mode: str = "3class") -> str:
    """Map an ECG annotation symbol to a standardized class.

    Args:
        symbol: MIT-BIH annotation symbol
        mode: ``'3class'`` (Normal, PVC, Other), ``'aami'`` (Normal, PVC, SVEB, Other),
              or ``'binary'`` (Normal vs Abnormal)

    Returns:
        Standardized class name or 'Unknown'
    """
    if not symbol or symbol in NON_BEAT_SYMBOLS:
        return "Unknown"

    if mode == "binary":
        return "Normal" if symbol in NORMAL_SYMBOLS else "Abnormal"

    if mode == "3class":
        if symbol in NORMAL_SYMBOLS:
            return "Normal"
        if symbol in PVC_SYMBOLS:
            return "PVC"
        if symbol in SVEB_SYMBOLS or symbol in OTHER_SYMBOLS:
            return "Other"
        return "Unknown"

    if symbol in NORMAL_SYMBOLS:
        return "Normal"
    if symbol in PVC_SYMBOLS:
        return "PVC"
    if symbol in SVEB_SYMBOLS:
        return "SVEB"
    if symbol in OTHER_SYMBOLS:
        return "Other"

    return "Unknown"


def map_symbol_to_binary(symbol: str) -> str:
    """Map an annotation symbol to the binary Normal-vs-Abnormal label."""
    return map_symbol_to_class(symbol, mode="binary")


def map_symbols_to_classes(
    symbols: Iterable[str],
    mode: str = "aami",
) -> List[str]:
    """Map a sequence of annotation symbols to standardized classes."""
    return [map_symbol_to_class(symbol, mode=mode) for symbol in symbols]


def get_class_distribution(symbols: Iterable[str], mode: str = "aami") -> Dict[str, int]:
    """Count class occurrences from annotation symbols."""
    counts: Dict[str, int] = {}
    for symbol in symbols:
        class_name = map_symbol_to_class(symbol, mode=mode)
        counts[class_name] = counts.get(class_name, 0) + 1
    return counts


def get_class_summary(symbols: Iterable[str], mode: str = "aami") -> Dict[str, float]:
    """Return class distribution as percentages."""
    counts = get_class_distribution(symbols, mode=mode)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {cls: count / total for cls, count in counts.items()}


def summarize_classes(symbols: Iterable[str], mode: str = "aami") -> str:
    """Create a human-readable summary of class distribution."""
    counts = get_class_distribution(symbols, mode=mode)
    if not counts:
        return "No annotations found"

    total = sum(counts.values())
    lines = ["Class Distribution Summary", "=" * 30]
    for cls, count in sorted(counts.items(), key=lambda x: -x[1]):
        percent = (count / total) * 100
        lines.append(f"{cls:15}: {count:4d} ({percent:5.1f}%)")
    return "\n".join(lines)