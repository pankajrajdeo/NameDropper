#!/bin/bash

# Setup PostgreSQL database for NameDropper
echo "🧬 Setting up NameDropper database..."

# Check if PostgreSQL is running
if ! pg_isready -q; then
    echo "❌ PostgreSQL is not running. Please start PostgreSQL first."
    echo "   On macOS: brew services start postgresql"
    echo "   On Ubuntu: sudo systemctl start postgresql"
    exit 1
fi

# Check if database exists
if psql -lqt | cut -d \| -f 1 | grep -qw ontology_mapper; then
    echo "⚠️  Database 'ontology_mapper' already exists."
    read -p "Do you want to drop and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Dropping existing database..."
        dropdb ontology_mapper
    else
        echo "✅ Using existing database."
        exit 0
    fi
fi

# Create database
echo "📊 Creating database 'ontology_mapper'..."
createdb ontology_mapper

# Install pgvector extension
echo "🔧 Installing pgvector extension..."
psql -d ontology_mapper -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Create basic schema
echo "🏗️  Creating basic schema..."
psql -d ontology_mapper << 'EOF'
-- Create ontology_terms table
CREATE TABLE IF NOT EXISTS ontology_terms (
    id SERIAL PRIMARY KEY,
    ontology_id VARCHAR(100) NOT NULL,
    ontology VARCHAR(50) NOT NULL,
    label TEXT NOT NULL,
    definition TEXT,
    synonyms TEXT[],
    embedding vector(384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_ontology_terms_ontology ON ontology_terms(ontology);
CREATE INDEX IF NOT EXISTS idx_ontology_terms_ontology_id ON ontology_terms(ontology_id);
CREATE INDEX IF NOT EXISTS idx_ontology_terms_label ON ontology_terms USING gin(to_tsvector('english', label));
CREATE INDEX IF NOT EXISTS idx_ontology_terms_embedding ON ontology_terms USING ivfflat (embedding vector_cosine_ops);

-- Create unique constraint
CREATE UNIQUE INDEX IF NOT EXISTS idx_ontology_terms_unique ON ontology_terms(ontology, ontology_id);
EOF

echo "✅ Database setup complete!"
echo "📊 Database 'ontology_mapper' is ready for use."
echo ""
echo "Next steps:"
echo "1. Copy env.example to .env and configure your database credentials"
echo "2. Install dependencies: pip install -r requirements.txt"
echo "3. Load ontologies: namedropper-manage load CL"
