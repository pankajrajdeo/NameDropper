#!/usr/bin/env python3
"""
Enhanced Metadata Harmonization CLI with Full Configuration Control

Features:
- Multiple search strategy selection and configuration
- Per-strategy top_k and threshold settings
- Configurable term expansion count
- Dynamic LLM prompt updates
- Production-level error handling and logging
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import yaml

# Import from the same package
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.ontology_mapper import UnifiedOntologyMapper
from core.harmonizer_agent import harmonize_metadata, HarmonizedTerm

from tqdm import tqdm

# Install and import Ollama embeddings
try:
    from langchain_ollama import OllamaEmbeddings
    OLLAMA_AVAILABLE = True
except ImportError:
    print("⚠️ langchain-ollama not installed. Installing...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "langchain-ollama"], check=True)
    from langchain_ollama import OllamaEmbeddings
    OLLAMA_AVAILABLE = True

@dataclass
class SearchStrategy:
    """Configuration for a single search strategy."""
    name: str
    enabled: bool = True
    top_k: int = 5
    threshold: float = 0.7
    priority: int = 0

@dataclass
class HarmonizationConfig:
    """Complete harmonization configuration."""
    strategies: Dict[str, SearchStrategy]
    term_expansion: bool = False
    expansion_count: int = 5
    dataset_description: str = ""
    field_mappings: Dict[str, str] = None
    ontology_filter: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class MetadataHarmonizer:
    """
    Enhanced metadata harmonizer with full configuration control.
    """
    
    def __init__(self, db_config: Dict = None):
        # Use default database configuration if none provided
        if db_config is None:
            db_config = {
                'db_host': os.getenv('DB_HOST', 'localhost'),
                'db_port': int(os.getenv('DB_PORT', '5432')),
                'db_name': os.getenv('DB_NAME', 'ontology_mapper'),
                'db_user': os.getenv('DB_USER', 'postgres'),
                'db_password': os.getenv('DB_PASSWORD', None),
                'embedding_service_url': None
            }
        
        self.mapper = UnifiedOntologyMapper(**db_config)
        print("✅ Connected to ontology database")
        
        # Initialize Ollama embeddings
        if OLLAMA_AVAILABLE:
            try:
                self.embeddings = OllamaEmbeddings(model="pankajrajdeo/biomed-embeddings-16l-fp16")
                print("✅ Ollama embeddings initialized with biomed-embeddings-16l-fp16")
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize Ollama embeddings: {e}")
                self.embeddings = None
        else:
            self.embeddings = None

    def parse_metadata_file(self, file_path: str) -> Dict[str, Any]:
        """Parse metadata file and extract field information."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract field information
            fields = {}
            if 'dataset' in data and '**Indexes and Annotations**' in data['dataset']:
                indexes = data['dataset']['**Indexes and Annotations**']
                for index_name, index_data in indexes.items():
                    if isinstance(index_data, dict):
                        for field_name, field_values in index_data.items():
                            if isinstance(field_values, list):
                                fields[field_name] = {
                                    'type': self._infer_field_type(field_name, field_values),
                                    'values': field_values,
                                    'count': len(field_values)
                                }
            
            return {
                'data': data,
                'fields': fields,
                'file_path': file_path
            }
        except Exception as e:
            print(f"❌ Error parsing metadata file: {e}")
            return None

    def _infer_field_type(self, field_name: str, values: List[str]) -> str:
        """Infer the field type based on field name and values."""
        field_name_lower = field_name.lower()
        
        # Cell type indicators
        if any(x in field_name_lower for x in ['cell', 'celltype', 'cell_type']):
            return 'cell_type'
        
        # Disease indicators
        if any(x in field_name_lower for x in ['disease', 'condition', 'phenotype']):
            return 'disease'
        
        # Tissue indicators
        if any(x in field_name_lower for x in ['tissue', 'organ', 'anatomy']):
            return 'tissue'
        
        # Chemical indicators
        if any(x in field_name_lower for x in ['chemical', 'compound', 'drug', 'metabolite']):
            return 'chemical'
        
        # Age indicators
        if any(x in field_name_lower for x in ['age', 'developmental', 'stage']):
            return 'age'
        
        # Sex indicators
        if any(x in field_name_lower for x in ['sex', 'gender']):
            return 'sex'
        
        # Default to other
        return 'other'

    def harmonize_field(self, field_name: str, field_data: Dict, config: HarmonizationConfig) -> Dict[str, Any]:
        """Harmonize all terms in a specific field."""
        print(f"\n🔍 Harmonizing field: {field_name} ({field_data['count']} terms)")
        
        harmonized_terms = []
        failed_terms = []
        
        # Create strategy configuration for harmonizer agent
        strategy_config = {
            'priority': [s.name for s in config.strategies.values() if s.enabled],
            'semantic_threshold': config.strategies.get('semantic', SearchStrategy('semantic')).threshold,
            'fuzzy_threshold': config.strategies.get('fuzzy', SearchStrategy('fuzzy')).threshold,
            'top_k': max(s.top_k for s in config.strategies.values() if s.enabled)
        }
        
        # Process each term
        for term in tqdm(field_data['values'], desc=f"Harmonizing {field_name}"):
            try:
                result = harmonize_metadata(
                    dataset_description=config.dataset_description,
                    term_name=term,
                    field_name=field_name,
                    mapper=self.mapper,
                    ontologies=config.ontology_filter,
                    k=strategy_config['top_k'],
                    use_query_expansion=config.term_expansion,
                    num_expanded_queries=config.expansion_count
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
                print(f"⚠️ Error harmonizing term '{term}': {e}")
                failed_terms.append({
                    'term': term,
                    'reason': f"Error: {str(e)}"
                })
        
        return {
            'field_name': field_name,
            'field_type': field_data['type'],
            'total_terms': field_data['count'],
            'harmonized_terms': harmonized_terms,
            'failed_terms': failed_terms,
            'success_rate': len(harmonized_terms) / field_data['count'] if field_data['count'] > 0 else 0
        }

    def harmonize_metadata_file(self, input_file: str, output_file: str, config: HarmonizationConfig) -> bool:
        """Harmonize an entire metadata file."""
        print(f"🚀 Starting harmonization of {input_file}")
        
        # Parse metadata file
        metadata_info = self.parse_metadata_file(input_file)
        if not metadata_info:
            return False
        
        # Extract dataset description if available
        if not config.dataset_description:
            if 'data' in metadata_info and 'dataset' in metadata_info['data']:
                dataset_meta = metadata_info['data']['dataset'].get('**Dataset Metadata**', {})
                config.dataset_description = dataset_meta.get('**Description**', 'No description available')
        
        print(f"📊 Dataset: {metadata_info['data']['dataset'].get('**Dataset Metadata', {}).get('**Dataset Name**', 'Unknown')}")
        print(f"📝 Description: {config.dataset_description[:100]}...")
        print(f"🔍 Fields found: {len(metadata_info['fields'])}")
        
        # Show field information
        for field_name, field_data in metadata_info['fields'].items():
            print(f"  • {field_name}: {field_data['type']} ({field_data['count']} terms)")
        
        # Harmonize each field
        harmonized_fields = {}
        total_terms = 0
        total_harmonized = 0
        
        for field_name, field_data in metadata_info['fields'].items():
            if field_data['count'] > 0:  # Only process fields with terms
                result = self.harmonize_field(field_name, field_data, config)
                harmonized_fields[field_name] = result
                total_terms += result['total_terms']
                total_harmonized += len(result['harmonized_terms'])
        
        # Create output structure
        output_data = {
            'input_file': input_file,
            'harmonization_config': config.to_dict(),
            'harmonization_summary': {
                'total_fields': len(harmonized_fields),
                'total_terms': total_terms,
                'total_harmonized': total_harmonized,
                'overall_success_rate': total_harmonized / total_terms if total_terms > 0 else 0,
                'timestamp': str(Path(input_file).stat().st_mtime)
            },
            'harmonized_fields': harmonized_fields,
            'original_metadata': metadata_info['data']
        }
        
        # Save harmonized data
        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"✅ Harmonized data saved to {output_file}")
            return True
        except Exception as e:
            print(f"❌ Error saving harmonized data: {e}")
            return False

    def close(self):
        """Close database connection."""
        if hasattr(self, 'mapper'):
            self.mapper.close()

def get_harmonization_config_from_user() -> HarmonizationConfig:
    """Interactive configuration setup."""
    print("🧬 --- Enhanced Metadata Harmonization Configuration ---")
    print("🔧 Using Ollama with pankajrajdeo/biomed-embeddings-16l-fp16 for embeddings")
    
    # Initialize strategies
    strategies = {
        'exact': SearchStrategy('exact', True, 5, 1.0, 1),
        'fuzzy': SearchStrategy('fuzzy', True, 5, 0.85, 2),
        'semantic': SearchStrategy('semantic', True, 5, 0.7, 3),
        'fulltext': SearchStrategy('fulltext', True, 5, 0.6, 4)
    }
    
    # 1. Strategy selection and configuration
    print("\n📋 --- Search Strategy Configuration ---")
    print("Available strategies: exact, fuzzy, semantic, fulltext")
    
    # Enable/disable strategies
    for strategy_name, strategy in strategies.items():
        while True:
            try:
                enabled = input(f"Enable {strategy_name} strategy? (y/n): ").strip().lower()
                if enabled in ['y', 'yes']:
                    strategy.enabled = True
                    break
                elif enabled in ['n', 'no']:
                    strategy.enabled = False
                    break
                else:
                    print("Please enter 'y' or 'n'")
            except (KeyboardInterrupt, EOFError):
                sys.exit("\nOperation cancelled by user.")
    
    # Configure enabled strategies
    enabled_strategies = [s for s in strategies.values() if s.enabled]
    if not enabled_strategies:
        print("❌ At least one strategy must be enabled!")
        sys.exit(1)
    
    print(f"\n🔧 Configuring {len(enabled_strategies)} enabled strategies:")
    
    for strategy in enabled_strategies:
        print(f"\n--- {strategy.name.upper()} Strategy ---")
        
        # Top-k configuration
        while True:
            try:
                top_k = input(f"Top-k for {strategy.name} (1-20, default {strategy.top_k}): ").strip()
                if not top_k:
                    break
                top_k_val = int(top_k)
                if 1 <= top_k_val <= 20:
                    strategy.top_k = top_k_val
                    break
                else:
                    print("Top-k must be between 1 and 20")
            except ValueError:
                print("Please enter a valid number")
            except (KeyboardInterrupt, EOFError):
                sys.exit("\nOperation cancelled by user.")
        
        # Threshold configuration
        if strategy.name in ['fuzzy', 'semantic']:
            while True:
                try:
                    threshold = input(f"Similarity threshold for {strategy.name} (0.0-1.0, default {strategy.threshold}): ").strip()
                    if not threshold:
                        break
                    threshold_val = float(threshold)
                    if 0.0 <= threshold_val <= 1.0:
                        strategy.threshold = threshold_val
                        break
                    else:
                        print("Threshold must be between 0.0 and 1.0")
                except ValueError:
                    print("Please enter a valid number")
                except (KeyboardInterrupt, EOFError):
                    sys.exit("\nOperation cancelled by user.")
    
    # 2. Term expansion configuration
    print("\n🔍 --- Term Expansion Configuration ---")
    while True:
        try:
            use_expansion = input("Enable AI-powered term expansion? (y/n): ").strip().lower()
            if use_expansion in ['y', 'yes']:
                term_expansion = True
                break
            elif use_expansion in ['n', 'no']:
                term_expansion = False
                break
            else:
                print("Please enter 'y' or 'n'")
        except (KeyboardInterrupt, EOFError):
            sys.exit("\nOperation cancelled by user.")
    
    expansion_count = 5
    if term_expansion:
        while True:
            try:
                count = input(f"Number of expanded terms (2-10, default {expansion_count}): ").strip()
                if not count:
                    break
                count_val = int(count)
                if 2 <= count_val <= 10:
                    expansion_count = count_val
                    break
                else:
                    print("Count must be between 2 and 10")
            except ValueError:
                print("Please enter a valid number")
            except (KeyboardInterrupt, EOFError):
                sys.exit("\nOperation cancelled by user.")
    
    # 3. Dataset description
    print("\n📝 --- Dataset Context ---")
    dataset_description = input("Enter dataset description (or press Enter to auto-detect): ").strip()
    
    # 4. Ontology filtering
    print("\n🎯 --- Ontology Selection ---")
    print("Available ontologies: CL, MONDO, UBERON, PATO, CHEBI, EFO, HPO, ORDO, etc.")
    ontology_filter = input("Enter comma-separated ontology codes (or press Enter for all): ").strip()
    if not ontology_filter:
        ontology_filter = None
    
    # Create configuration
    config = HarmonizationConfig(
        strategies=strategies,
        term_expansion=term_expansion,
        expansion_count=expansion_count,
        dataset_description=dataset_description,
        ontology_filter=ontology_filter
    )
    
    return config

def main():
    """Main function to configure and run the harmonization process."""
    config = get_harmonization_config_from_user()
    
    # Display configuration summary
    print("\n📋 --- Configuration Summary ---")
    print(f"  Enabled Strategies: {', '.join(s.name for s in config.strategies.values() if s.enabled)}")
    print(f"  Term Expansion: {'Yes' if config.term_expansion else 'No'}")
    if config.term_expansion:
        print(f"  Expansion Count: {config.expansion_count}")
    print(f"  Ontology Filter: {config.ontology_filter or 'All ontologies'}")
    
    # Show strategy details
    print("\n  Strategy Details:")
    for strategy in config.strategies.values():
        if strategy.enabled:
            print(f"    • {strategy.name}: top_k={strategy.top_k}, threshold={strategy.threshold}")
    
    print("------------------------\n")
    
    try:
        confirm = input("Start harmonization with this configuration? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Operation cancelled.")
            return
    except (KeyboardInterrupt, EOFError):
        sys.exit("\nOperation cancelled by user.")

    # Get all metadata files in the current working directory
    metadata_dir = Path.cwd()
    metadata_files = list(metadata_dir.glob("*_metadata.json"))
    
    if not metadata_files:
        print(f"❌ No metadata files found in {metadata_dir}")
        print("💡 Make sure you're in the directory containing your metadata files")
        print("   Files should end with '_metadata.json'")
        return
    
    print(f"\n📁 Found {len(metadata_files)} metadata files:")
    for i, file in enumerate(metadata_files):
        print(f"  {i+1}. {file.name}")
    
    # Let user choose which files to process
    try:
        choice = input(f"\nEnter file number to process (1-{len(metadata_files)}) or 'all' for all files: ").strip()
        
        if choice.lower() == 'all':
            files_to_process = metadata_files
        else:
            file_num = int(choice)
            if 1 <= file_num <= len(metadata_files):
                files_to_process = [metadata_files[file_num - 1]]
            else:
                print("Invalid file number.")
                return
    except ValueError:
        print("Invalid input. Please enter a number or 'all'.")
        return
    except (KeyboardInterrupt, EOFError):
        sys.exit("\nOperation cancelled by user.")

    print("\n🚀 Initializing harmonizer...")
    harmonizer = MetadataHarmonizer()
    
    try:
        for input_file in files_to_process:
            print(f"\n{'='*60}")
            print(f"Processing: {input_file.name}")
            print(f"{'='*60}")
            
            # Generate output filename
            output_file = input_file.parent / f"{input_file.stem}_harmonized_enhanced.json"
            
            print(f"Starting harmonization process...")
            success = harmonizer.harmonize_metadata_file(str(input_file), str(output_file), config)
            
            if success:
                print(f"✅ Completed: {input_file.name} → {output_file.name}")
            else:
                print(f"❌ Failed: {input_file.name}")
            
    except Exception as e:
        print(f"❌ Error during harmonization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        harmonizer.close()
        print("\n🔒 Database connection closed.")

if __name__ == "__main__":
    main() 