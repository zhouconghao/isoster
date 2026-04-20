"""Dataset adapters. Each adapter maps a dataset-specific on-disk layout
onto the common ``DatasetAdapter`` protocol so the orchestrator can iterate
galaxies without knowing about file-naming conventions.
"""

from .base import DatasetAdapter, GalaxyBundle, GalaxyMetadata

__all__ = ["DatasetAdapter", "GalaxyBundle", "GalaxyMetadata"]
