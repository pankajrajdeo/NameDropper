# NameDropper 🧬

Biomedical ontology harmonization system with AI-powered mapping and a web interface for testing.

## 🚀 Quick Start

### 1. Setup Database
```bash
./scripts/setup_database.sh
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
cp env.example .env
# Edit .env with your database credentials
```

### 4. Load Ontologies
```bash
namedropper-manage load CL
namedropper-manage load MONDO
namedropper-manage rebuild-indexes
```

### 5. Run Web Interface (Testing)
```bash
namedropper-web
```

### 6. Process Metadata (Production)
```bash
namedropper --input metadata.json --output harmonized.json
```

## 🏗️ Architecture

- **Core Engine**: PostgreSQL-based ontology mapper with pgvector
- **AI Harmonization**: LLM-powered term matching using GPT-4o-mini
- **Local Embeddings**: Ollama integration with biomedical embeddings model
- **Web Interface**: Gradio-based testing and demonstration interface
- **CLI Tools**: Command-line tools for database management and batch processing

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Usage Examples](docs/usage.md)
- [API Reference](docs/api.md)
- [Database Setup](docs/database.md)

## 🔧 Development

```bash
# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/

# Format code
black src/
```

## 📊 Supported Ontologies

- **Clinical**: HPO, MONDO, DOID, ORDO, MAXO, OGMS
- **Cellular**: CL, UBERON, FMA, RADLEX
- **Chemical**: CHEBI, GO, PR, SO
- **Experimental**: EFO, OBI, PATO, NCBITAXON

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.
