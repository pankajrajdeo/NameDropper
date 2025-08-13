# Installation Guide

## Prerequisites

### 1. Python Environment
- Python 3.8 or higher
- pip package manager

### 2. PostgreSQL Database
- PostgreSQL 12 or higher
- pgvector extension installed

### 3. Ollama
- Ollama installed and running locally
- `pankajrajdeo/biomed-embeddings-16l-fp16` model pulled

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/namedropper.git
cd namedropper
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install in Development Mode
```bash
pip install -e .
```

### 5. Setup Database
```bash
./scripts/setup_database.sh
```

### 6. Configure Environment
```bash
cp env.example .env
# Edit .env with your database credentials
```

### 7. Pull Ollama Model
```bash
ollama pull pankajrajdeo/biomed-embeddings-16l-fp16
```

## Verification

### Test Installation
```bash
python -c "import namedropper; print('✅ Installation successful!')"
```

### Test Database Connection
```bash
namedropper-manage --help
```

### Test Web Interface
```bash
namedropper-web
```

## Troubleshooting

### Common Issues

1. **PostgreSQL Connection Failed**
   - Ensure PostgreSQL is running
   - Check credentials in `.env` file
   - Verify database exists

2. **pgvector Extension Missing**
   - Install pgvector: `CREATE EXTENSION vector;`
   - Restart PostgreSQL

3. **Ollama Not Available**
   - Start Ollama: `ollama serve`
   - Pull required model: `ollama pull pankajrajdeo/biomed-embeddings-16l-fp16`

4. **Import Errors**
   - Ensure you're in the virtual environment
   - Reinstall in development mode: `pip install -e .`
