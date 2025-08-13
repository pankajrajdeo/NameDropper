#!/usr/bin/env python3
"""
Example: Metadata Harmonization with NameDropper

This script demonstrates how to use NameDropper to harmonize
metadata terms against biomedical ontologies.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cli.harmonize_metadata import MetadataHarmonizer

def example_cell_type_harmonization():
    """Example: Harmonize cell type terms."""
    print("🧬 Example: Cell Type Harmonization")
    print("=" * 50)
    
    # Initialize harmonizer
    harmonizer = MetadataHarmonizer()
    
    # Define harmonization strategy
    strategy = {
        'priority': ['exact', 'fuzzy', 'semantic'],
        'semantic_threshold': 0.7
    }
    
    # Sample cell type terms
    cell_types = [
        "AT1",
        "AT2", 
        "macrophage",
        "fibroblast",
        "endothelial cell",
        "smooth muscle cell"
    ]
    
    print(f"Processing {len(cell_types)} cell type terms...")
    print()
    
    for term in cell_types:
        result = harmonizer.harmonize_term(term, strategy)
        
        if result['harmonized_term']:
            print(f"✅ {term:20} -> {result['harmonized_term']:30} ({result['ontology']}:{result['ontology_id']})")
            print(f"   Match type: {result['match_type']}, Score: {result['score']}")
        else:
            print(f"❌ {term:20} -> No match found")
        
        print()
    
    harmonizer.close()

def example_disease_harmonization():
    """Example: Harmonize disease terms."""
    print("🏥 Example: Disease Harmonization")
    print("=" * 50)
    
    # Initialize harmonizer
    harmonizer = MetadataHarmonizer()
    
    # Define harmonization strategy
    strategy = {
        'priority': ['exact', 'fuzzy', 'semantic'],
        'semantic_threshold': 0.6
    }
    
    # Sample disease terms
    diseases = [
        "pulmonary fibrosis",
        "COPD",
        "asthma",
        "lung cancer",
        "pneumonia",
        "tuberculosis"
    ]
    
    print(f"Processing {len(diseases)} disease terms...")
    print()
    
    for disease in diseases:
        result = harmonizer.harmonize_term(disease, strategy)
        
        if result['harmonized_term']:
            print(f"✅ {disease:20} -> {result['harmonized_term']:30} ({result['ontology']}:{result['ontology_id']})")
            print(f"   Match type: {result['match_type']}, Score: {result['score']}")
        else:
            print(f"❌ {disease:20} -> No match found")
        
        print()
    
    harmonizer.close()

def example_tissue_harmonization():
    """Example: Harmonize tissue terms."""
    print("🫁 Example: Tissue Harmonization")
    print("=" * 50)
    
    # Initialize harmonizer
    harmonizer = MetadataHarmonizer()
    
    # Define harmonization strategy
    strategy = {
        'priority': ['exact', 'fuzzy', 'semantic'],
        'semantic_threshold': 0.7
    }
    
    # Sample tissue terms
    tissues = [
        "lung",
        "lung parenchyma",
        "bronchus",
        "alveoli",
        "pleura",
        "mediastinum"
    ]
    
    print(f"Processing {len(tissues)} tissue terms...")
    print()
    
    for tissue in tissues:
        result = harmonizer.harmonize_term(tissue, strategy)
        
        if result['harmonized_term']:
            print(f"✅ {tissue:20} -> {result['harmonized_term']:30} ({result['ontology']}:{result['ontology_id']})")
            print(f"   Match type: {result['match_type']}, Score: {result['score']}")
        else:
            print(f"❌ {tissue:20} -> No match found")
        
        print()
    
    harmonizer.close()

def main():
    """Run all examples."""
    print("🚀 NameDropper Examples")
    print("=" * 60)
    print()
    
    try:
        # Run examples
        example_cell_type_harmonization()
        print()
        example_disease_harmonization()
        print()
        example_tissue_harmonization()
        
        print("🎉 All examples completed successfully!")
        print("\n💡 Tips:")
        print("   - Adjust semantic_threshold for different sensitivity levels")
        print("   - Use different priority orders for different use cases")
        print("   - Check the web interface with 'namedropper-web' for interactive testing")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Ensure database is running and accessible")
        print("   2. Check environment variables in .env file")
        print("   3. Verify ontologies are loaded: namedropper-manage stats")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
