# Usage Guide

## Quick Start

### 1. Load Ontologies
```bash
# Load individual ontologies
namedropper-manage load CL
namedropper-manage load MONDO
namedropper-manage load HPO

# Load all ontologies
namedropper-manage load-all

# Rebuild indexes for performance
namedropper-manage rebuild-indexes
```

### 2. Test with Web Interface
```bash
namedropper-web
```
Open browser to `http://localhost:7862`

### 3. Process Metadata Files
```bash
namedropper --input metadata.json --output harmonized.json
```

## Command Line Tools

### namedropper-manage
Database management and ontology operations.

```bash
# Load ontologies
namedropper-manage load CL
namedropper-manage load MONDO

# Database operations
namedropper-manage rebuild-indexes
namedropper-manage vacuum
namedropper-manage stats

# Remove ontologies
namedropper-manage remove CL
```

### namedropper
Batch metadata harmonization.

```bash
# Basic usage
namedropper --input metadata.json --output harmonized.json

# With specific ontologies
namedropper --input metadata.json --output harmonized.json --ontologies CL,MONDO

# With custom strategy
namedropper --input metadata.json --output harmonized.json --strategy exact,fuzzy,semantic
```

### namedropper-web
Launch the Gradio web interface.

```bash
namedropper-web
```

## Configuration

### Environment Variables
```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ontology_mapper
DB_USER=postgres
DB_PASSWORD=your_password

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=pankajrajdeo/biomed-embeddings-16l-fp16
```

### Strategy Configuration
```python
strategy = {
    'priority': ['exact', 'fuzzy', 'semantic', 'fulltext'],
    'semantic_threshold': 0.7,
    'fuzzy_threshold': 0.85
}
```

## API Usage

### Python API
```python
from namedropper import MetadataHarmonizer, UnifiedOntologyMapper

# Initialize harmonizer
harmonizer = MetadataHarmonizer()

# Harmonize single term
result = harmonizer.harmonize_term("AT1", strategy)

# Process file
harmonizer.harmonize_metadata_file("input.json", "output.json", strategy)
```

### Search Methods
```python
mapper = UnifiedOntologyMapper()

# Exact match
exact_matches = mapper.exact_match("diabetes")

# Fuzzy match
fuzzy_matches = mapper.fuzzy_match("diabetes", threshold=0.8)

# Semantic match
semantic_matches = mapper.embedding_match("diabetes", threshold=0.7)

# Full-text search
fulltext_matches = mapper.fulltext_match("diabetes")
```

## Examples

### Cell Type Harmonization
```python
from namedropper import MetadataHarmonizer

harmonizer = MetadataHarmonizer()
strategy = {'priority': ['exact', 'fuzzy', 'semantic']}

# Harmonize cell types
cell_types = ["AT1", "AT2", "macrophage"]
for cell_type in cell_types:
    result = harmonizer.harmonize_term(cell_type, strategy)
    print(f"{cell_type} -> {result['harmonized_term']}")
```

### Disease Harmonization
```python
# Harmonize diseases
diseases = ["pulmonary fibrosis", "COPD", "asthma"]
for disease in diseases:
    result = harmonizer.harmonize_term(disease, strategy)
    print(f"{disease} -> {result['ontology_id']}")
```

## Output Format

### Harmonized Results
```json
{
  "original_term": "AT1",
  "harmonized_term": "alveolar type I cell",
  "ontology_id": "CL:0002062",
  "ontology": "CL",
  "match_type": "exact",
  "score": 100,
  "definition": "A squamous epithelial cell...",
  "similarity": 1.0
}
```

### File Output
```json
{
  "dataset": {
    "**Indexes and Annotations**": {
      // Original annotations preserved
    },
    "**Harmonized Annotations**": {
      "ann_finest_level_harmonized": [...],
      "disease_harmonized": [...],
      "tissue_harmonized": [...]
    }
  }
}
```
