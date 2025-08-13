#!/usr/bin/env python3
"""
Production-Level Batch Metadata Harmonizer

Features:
- YAML configuration files
- Batch processing of multiple files
- Field-specific ontology targeting
- Comprehensive logging and reporting
- Performance optimization
"""

import argparse
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys

# Import from the same package
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.ontology_mapper import UnifiedOntologyMapper
from core.harmonizer_agent import harmonize_metadata, HarmonizedTerm

class ProductionHarmonizer:
    """Production-level metadata harmonizer with full configuration support."""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load config first (without logging)
        self.config = self._load_config_silent(config_path)
        # Setup logging with config
        self._setup_logging()
        # Setup database
        self._setup_database()
        
    def _load_config_silent(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file or use defaults (without logging)."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = self._get_default_config()
        
        return config
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file or use defaults."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.logger.info(f"Loaded configuration from {config_path}")
        else:
            config = self._get_default_config()
            self.logger.info("Using default configuration")
        
        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'strategies': {
                'exact': {'enabled': True, 'top_k': 5, 'threshold': 1.0, 'priority': 1},
                'fuzzy': {'enabled': True, 'top_k': 5, 'threshold': 0.85, 'priority': 2},
                'semantic': {'enabled': True, 'top_k': 5, 'threshold': 0.7, 'priority': 3},
                'fulltext': {'enabled': False, 'top_k': 5, 'threshold': 0.6, 'priority': 4}
            },
            'term_expansion': {'enabled': False, 'expansion_count': 5},
            'ontology_filter': None,
            'field_mappings': {},
            'output': {
                'format': 'json',
                'include_original': True,
                'include_reasoning': True,
                'include_confidence_scores': True,
                'include_alternative_matches': True
            },
            'performance': {
                'batch_size': 100,
                'max_workers': 1,
                'timeout': 300,
                'retry_attempts': 3
            },
            'logging': {
                'level': 'INFO',
                'file': 'harmonization.log',
                'include_timestamps': True,
                'include_performance_metrics': True
            }
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_file = log_config.get('file', 'harmonization.log')
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Production Harmonizer initialized")
    
    def _setup_database(self):
        """Setup database connection."""
        try:
            self.mapper = UnifiedOntologyMapper()
            self.logger.info("✅ Database connection established")
        except Exception as e:
            self.logger.error(f"❌ Database connection failed: {e}")
            raise
    
    def _get_field_ontology_filter(self, field_name: str) -> Optional[str]:
        """Get ontology filter for a specific field."""
        field_mappings = self.config.get('field_mappings', {})
        field_type = self._infer_field_type(field_name)
        
        # Check field-specific mapping first
        if field_type in field_mappings:
            return field_mappings[field_type]
        
        # Fall back to global ontology filter
        return self.config.get('ontology_filter')
    
    def _infer_field_type(self, field_name: str) -> str:
        """Infer field type from field name."""
        field_name_lower = field_name.lower()
        
        if any(x in field_name_lower for x in ['cell', 'celltype', 'cell_type']):
            return 'cell_type'
        elif any(x in field_name_lower for x in ['disease', 'condition', 'phenotype']):
            return 'disease'
        elif any(x in field_name_lower for x in ['tissue', 'organ', 'anatomy']):
            return 'tissue'
        elif any(x in field_name_lower for x in ['chemical', 'compound', 'drug', 'metabolite']):
            return 'chemical'
        elif any(x in field_name_lower for x in ['age', 'developmental', 'stage']):
            return 'age'
        elif any(x in field_name_lower for x in ['sex', 'gender']):
            return 'sex'
        else:
            return 'other'
    
    def harmonize_field(self, field_name: str, field_values: List[str], 
                       dataset_description: str) -> Dict[str, Any]:
        """Harmonize a single field with all its terms."""
        self.logger.info(f"🔍 Harmonizing field: {field_name} ({len(field_values)} terms)")
        
        # Get field-specific ontology filter
        ontology_filter = self._get_field_ontology_filter(field_name)
        if ontology_filter:
            self.logger.info(f"🎯 Using ontology filter for {field_name}: {ontology_filter}")
        
        # Get strategy configuration
        strategies = self.config.get('strategies', {})
        enabled_strategies = [s for s in strategies.values() if s.get('enabled', False)]
        
        if not enabled_strategies:
            raise ValueError("No strategies enabled in configuration")
        
        # Get term expansion configuration
        term_expansion = self.config.get('term_expansion', {})
        use_expansion = term_expansion.get('enabled', False)
        expansion_count = term_expansion.get('expansion_count', 5)
        
        # Get performance configuration
        performance = self.config.get('performance', {})
        batch_size = performance.get('batch_size', 100)
        top_k = max(s.get('top_k', 5) for s in enabled_strategies)
        
        harmonized_terms = []
        failed_terms = []
        
        # Process terms in batches
        for i in range(0, len(field_values), batch_size):
            batch = field_values[i:i + batch_size]
            self.logger.info(f"📦 Processing batch {i//batch_size + 1}/{(len(field_values) + batch_size - 1)//batch_size}")
            
            for term in batch:
                try:
                    result = harmonize_metadata(
                        dataset_description=dataset_description,
                        term_name=term,
                        field_name=field_name,
                        mapper=self.mapper,
                        ontologies=ontology_filter,
                        k=top_k,
                        use_query_expansion=use_expansion,
                        num_expanded_queries=expansion_count
                    )
                    
                    if result.status == "success":
                        harmonized_terms.append({
                            'original_term': term,
                            'harmonized_term': result.best_match.label,
                            'ontology_id': result.best_match.ontology_id,
                            'ontology': result.best_match.ontology,
                            'match_type': result.method,
                            'score': int(result.confidence * 100),
                            'definition': result.best_match.definition,
                            'similarity': result.best_match.similarity,
                            'reasoning': result.reasoning
                        })
                    else:
                        failed_terms.append({
                            'term': term,
                            'reason': result.reasoning
                        })
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Error harmonizing term '{term}': {e}")
                    failed_terms.append({
                        'term': term,
                        'reason': f"Error: {str(e)}"
                    })
        
        return {
            'field_name': field_name,
            'field_type': self._infer_field_type(field_name),
            'total_terms': len(field_values),
            'harmonized_terms': harmonized_terms,
            'failed_terms': failed_terms,
            'success_rate': len(harmonized_terms) / len(field_values) if field_values else 0,
            'ontology_filter_used': ontology_filter
        }
    
    def harmonize_file(self, input_file: str, output_file: str, 
                       dataset_description: str = "") -> bool:
        """Harmonize a single metadata file."""
        self.logger.info(f"🚀 Starting harmonization of {input_file}")
        
        try:
            # Parse metadata file
            with open(input_file, 'r') as f:
                metadata = json.load(f)
            
            # Extract fields to harmonize
            fields_to_harmonize = {}
            if 'dataset' in metadata and '**Indexes and Annotations**' in metadata['dataset']:
                indexes = metadata['dataset']['**Indexes and Annotations**']
                for index_name, index_data in indexes.items():
                    if isinstance(index_data, dict):
                        for field_name, field_values in index_data.items():
                            if isinstance(field_values, list) and len(field_values) > 0:
                                fields_to_harmonize[field_name] = field_values
            
            if not fields_to_harmonize:
                self.logger.warning("⚠️ No fields found to harmonize")
                return False
            
            # Extract dataset description if not provided
            if not dataset_description:
                if 'dataset' in metadata and '**Dataset Metadata**' in metadata['dataset']:
                    dataset_meta = metadata['dataset']['**Dataset Metadata**']
                    dataset_description = dataset_meta.get('**Description**', 'No description available')
            
            self.logger.info(f"📊 Dataset: {metadata['dataset'].get('**Dataset Metadata', {}).get('**Dataset Name**', 'Unknown')}")
            self.logger.info(f"📝 Description: {dataset_description[:100]}...")
            self.logger.info(f"🔍 Fields found: {len(fields_to_harmonize)}")
            
            # Show field information
            for field_name, field_values in fields_to_harmonize.items():
                self.logger.info(f"  • {field_name}: {self._infer_field_type(field_name)} ({len(field_values)} terms)")
            
            # Harmonize each field
            harmonized_fields = {}
            total_terms = 0
            total_harmonized = 0
            
            for field_name, field_values in fields_to_harmonize.items():
                result = self.harmonize_field(field_name, field_values, dataset_description)
                harmonized_fields[field_name] = result
                total_terms += result['total_terms']
                total_harmonized += len(result['harmonized_terms'])
            
            # Create output structure
            output_data = {
                'input_file': input_file,
                'harmonization_config': self.config,
                'harmonization_summary': {
                    'total_fields': len(harmonized_fields),
                    'total_terms': total_terms,
                    'total_harmonized': total_harmonized,
                    'overall_success_rate': total_harmonized / total_terms if total_terms > 0 else 0,
                    'timestamp': datetime.now().isoformat(),
                    'performance_metrics': {
                        'batch_size': self.config.get('performance', {}).get('batch_size', 100),
                        'strategies_used': [s for s in self.config.get('strategies', {}).values() if s.get('enabled', False)]
                    }
                },
                'harmonized_fields': harmonized_fields
            }
            
            # Include original metadata if configured
            if self.config.get('output', {}).get('include_original', True):
                output_data['original_metadata'] = metadata
            
            # Save harmonized data
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            self.logger.info(f"✅ Harmonized data saved to {output_file}")
            self.logger.info(f"📊 Summary: {total_harmonized}/{total_terms} terms harmonized ({total_harmonized/total_terms*100:.1f}% success)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error harmonizing file {input_file}: {e}")
            return False
    
    def close(self):
        """Close database connection."""
        if hasattr(self, 'mapper'):
            self.mapper.close()
            self.logger.info("🔒 Database connection closed")

def main():
    """Main function for batch harmonization."""
    parser = argparse.ArgumentParser(description="Production Batch Metadata Harmonizer")
    parser.add_argument("--config", "-c", help="Path to YAML configuration file")
    parser.add_argument("--input", "-i", required=True, help="Input metadata file or directory")
    parser.add_argument("--output", "-o", help="Output directory (default: same as input)")
    parser.add_argument("--description", "-d", help="Dataset description (overrides config)")
    
    args = parser.parse_args()
    
    try:
        # Initialize harmonizer
        harmonizer = ProductionHarmonizer(args.config)
        
        input_path = Path(args.input)
        output_dir = Path(args.output) if args.output else input_path.parent
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process input
        if input_path.is_file():
            # Single file
            output_file = output_dir / f"{input_path.stem}_harmonized_production.json"
            success = harmonizer.harmonize_file(str(input_path), str(output_file), args.description)
            if success:
                print(f"✅ Successfully harmonized: {input_path.name}")
            else:
                print(f"❌ Failed to harmonize: {input_path.name}")
                
        elif input_path.is_dir():
            # Directory of files
            metadata_files = list(input_path.glob("*_metadata.json"))
            if not metadata_files:
                print(f"❌ No metadata files found in {input_path}")
                return
            
            print(f"📁 Found {len(metadata_files)} metadata files")
            
            for metadata_file in metadata_files:
                output_file = output_dir / f"{metadata_file.stem}_harmonized_production.json"
                print(f"\n🚀 Processing: {metadata_file.name}")
                
                success = harmonizer.harmonize_file(str(metadata_file), str(output_file), args.description)
                if success:
                    print(f"✅ Completed: {metadata_file.name}")
                else:
                    print(f"❌ Failed: {metadata_file.name}")
        else:
            print(f"❌ Input path does not exist: {input_path}")
            return
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'harmonizer' in locals():
            harmonizer.close()

if __name__ == "__main__":
    main()
