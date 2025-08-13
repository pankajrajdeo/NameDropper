"""
NameDropper - Biomedical ontology harmonization system
"""

__version__ = "0.1.0"
__author__ = "NameDropper Contributors"
__email__ = "your.email@example.com"

from .core.ontology_mapper import UnifiedOntologyMapper
from .core.harmonizer_agent import harmonize_metadata, HarmonizedTerm
from .cli.harmonize_metadata import MetadataHarmonizer

__all__ = [
    "UnifiedOntologyMapper",
    "harmonize_metadata", 
    "HarmonizedTerm",
    "MetadataHarmonizer"
]
