"""
Dataset Format Converters
=========================
Converts raw dataset records into unified StandardizedECGRecord entities.
"""

from src.datasets.converters.mit_bih_converter import convert_mit_record_to_standard

__all__ = ["convert_mit_record_to_standard"]
