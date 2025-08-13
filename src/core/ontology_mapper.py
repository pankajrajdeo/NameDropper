#!/usr/bin/env python3
"""
Unified Ontology Mapper with PostgreSQL Backend

Comprehensive mapper for multiple ontologies with:
- Direct PostgreSQL storage and processing
- Exact/fuzzy matching using database indices
- Embedding-based matching using pgvector
- Full-text search with ranking
- Unified search across all ontologies
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import argparse
import os
import requests
import time
from tqdm import tqdm
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
import numpy as np

# Required dependencies
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, execute_values
except ImportError:
    print("Installing psycopg2-binary...")
    import subprocess
    subprocess.run(["pip", "install", "psycopg2-binary"])
    import psycopg2
    from psycopg2.extras import RealDictCursor, execute_values

try:
    import numpy as np
except ImportError:
    print("Installing numpy...")
    import subprocess
    subprocess.run(["pip", "install", "numpy"])
    import numpy as np

# Ollama embedding service configuration (local)
EMBEDDING_SERVICE_URL = None


class UnifiedOntologyMapper:
    """PostgreSQL-based unified ontology mapper with advanced search capabilities."""
    
    def __init__(self, 
                 db_host='localhost', 
                 db_port=5432, 
                 db_name='ontology_mapper',
                 db_user='postgres', 
                 db_password=None,
                 embedding_service_url=EMBEDDING_SERVICE_URL,
                 auto_create_schema=True,
                 max_text_length=1500):
        
        self.embedding_service_url = embedding_service_url
        self.embedding_service_available = False
        self.max_text_length = max_text_length
        
        # XML namespaces for OWL parsing
        self.namespaces = {
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
            'owl': 'http://www.w3.org/2002/07/owl#',
            'obo': 'http://purl.obolibrary.org/obo/',
            'oboInOwl': 'http://www.geneontology.org/formats/oboInOwl#',
            'dc': 'http://purl.org/dc/elements/1.1/',
            'skos': 'http://www.w3.org/2004/02/skos/core#',
        }
        
        # Ontology configuration
        self.ontology_configs = {
            'PATO': {
                'file': 'data/ontologies/pato.owl',
                'id_pattern': r'http://purl\.obolibrary\.org/obo/(PATO_\d+)',
                'name': 'Phenotype and Trait Ontology'
            },
            'UBERON': {
                'file': 'data/ontologies/uberon.owl', 
                'id_pattern': r'http://purl\.obolibrary\.org/obo/(UBERON_\d+)',
                'name': 'Anatomy Ontology'
            },
            'CL': {
                'file': 'data/ontologies/cl.owl',
                'id_pattern': r'http://purl\.obolibrary\.org/obo/(CL_\d+)',
                'name': 'Cell Ontology'
            },
            'EFO': {
                'file': 'data/ontologies/efo.owl',
                'id_pattern': r'http://www\.ebi\.ac\.uk/efo/(EFO_\d+)',
                'name': 'Experimental Factor Ontology'
            },
            'HANCESTRO': {
                'file': 'data/ontologies/hancestro-base.owl',
                'id_pattern': r'http://purl\.obolibrary\.org/obo/(HANCESTRO_\d+)',
                'name': 'Human Ancestry Ontology'
            },
            'HSAPDV': {
                'file': 'data/ontologies/hsapdv.owl',
                'id_pattern': r'http://purl\.obolibrary\.org/obo/(HsapDv_\d+)',
                'name': 'Human Developmental Stages'
            },
            'MONDO': {
                'file': 'data/ontologies/mondo.owl',
                'id_pattern': r'http://purl\.obolibrary\.org/obo/(MONDO_\d+)',
                'name': 'Disease Ontology'
            },
            'MMUSDV': {
                'file': 'data/ontologies/mmusdv.owl',
                'id_pattern': r'http://purl\.obolibrary\.org/obo/(MmusDv_\d+)',
                'name': 'Mouse Developmental Stages'
            },
            'NCBITAXON': {
                'file': 'data/ontologies/ncbitaxon.owl',
                'id_pattern': r'http://purl\.obolibrary\.org/obo/(NCBITaxon_\d+)',
                'name': 'NCBI Organismal Classification'
            },
            'HPO': {
                'file': 'data/ontologies/hpo.owl',
                'id_pattern': r'http://purl\.obolibrary\.org/obo/(HP_\d+)',
                'name': 'Human Phenotype Ontology'
            },
            'CHEBI': {
                'file': 'data/ontologies/chebi.owl',
                'id_pattern': r'http://purl\.obolibrary\.org/obo/(CHEBI_\d+)',
                'name': 'Chemical Entities of Biological Interest'
            },
            'SO': {
                'file': 'data/ontologies/so.owl',
                'id_pattern': r'http://purl\.obolibrary\.org/obo/(SO_\d+)',
                'name': 'Sequence Ontology'
            },
            'OBI': {
                'file': 'data/ontologies/obi.owl',
                'id_pattern': r'http://purl\.obolibrary\.org/obo/(OBI_\d+)',
                'name': 'Ontology for Biomedical Investigations'
            },
            'ORDO': {
                'file': 'data/ontologies/ordo.owl',
                'id_pattern': r'http://www\.orpha\.net/ORDO/(Orphanet_\d+)',
                'name': 'Orphanet Rare Disease Ontology'
            },
            'PR': {
                'file': 'data/ontologies/pr.owl',
                'id_pattern': r'http://purl\.obolibrary\.org/obo/(PR_\d+)',
                'name': 'Protein Ontology'
            },
            'GENO': {
                'file': 'data/ontologies/geno.owl',
                'id_pattern': r'http://purl\.obolibrary\.org/obo/(GENO_\d+)',
                'name': 'Genotype Ontology'
            },
            'DOID': {
                'file': 'data/ontologies/doid.owl',
                'id_pattern': r'http://purl\.obolibrary\.org/obo/(DOID_\d+)',
                'name': 'Disease Ontology'
            },
            'GSSO': {
                'file': 'data/ontologies/gsso.owl',
                'id_pattern': r'http://purl\.obolibrary\.org/obo/(GSSO_\d+)',
                'name': 'Gender, Sex, and Sexual Orientation Ontology'
            },
            'MAXO': {
                'file': 'data/ontologies/maxo.owl',
                'id_pattern': r'http://purl\.obolibrary\.org/obo/(MAXO_\d+)',
                'name': 'Medical Action Ontology'
            },
            'OGMS': {
                'file': 'data/ontologies/ogms.owl',
                'id_pattern': r'http://purl\.obolibrary\.org/obo/(OGMS_\d+)',
                'name': 'Ontology for General Medical Science'
            },
            'VO': {
                'file': 'data/ontologies/vo.owl',
                'id_pattern': r'http://purl\.obolibrary\.org/obo/(VO_\d+)',
                'name': 'Vaccine Ontology'
            },
            'GO': {
                'file': 'data/ontologies/go.owl',
                'id_pattern': r'http://purl\.obolibrary\.org/obo/(GO_\d+)',
                'name': 'Gene Ontology'
            },
            'RADLEX': {
                'file': 'data/ontologies/radlex.owl',
                'id_pattern': r'http://www\.ebi\.ac\.uk/efo/(RID\d+)',
                'name': 'RadLex Radiology Lexicon'
            },
            'MPATH': {
                'file': 'data/ontologies/mpath.owl',
                'id_pattern': r'http://purl\.obolibrary\.org/obo/(MPATH_\d+)',
                'name': 'Molecular Pathways Ontology'
            },
            'FMA': {
                'file': 'data/ontologies/fma.owl',
                'id_pattern': r'http://purl\.org/sig/ont/fma/(fma\d+)',
                'name': 'Foundational Model of Anatomy'
            }     
        }
        
        # Connect to PostgreSQL
        try:
            self.conn = psycopg2.connect(
                host=db_host,
                port=db_port,
                database=db_name,
                user=db_user,
                password=db_password,
                cursor_factory=RealDictCursor
            )
            print(f"✅ Connected to PostgreSQL at {db_host}:{db_port}")
        except Exception as e:
            print(f"❌ Failed to connect to PostgreSQL: {e}")
            print(f"Make sure database '{db_name}' exists: createdb {db_name}")
            raise e
        
        # Test embedding service
        self._test_embedding_service()
        
        # Create schema if requested
        if auto_create_schema:
            self._ensure_schema_exists()

    def _test_embedding_service(self):
        """Test if embedding services are available."""
        # Check for Ollama embeddings first
        try:
            from langchain_ollama import OllamaEmbeddings
            test_embeddings = OllamaEmbeddings(model="pankajrajdeo/biomed-embeddings-16l-fp16")
            # Test with a simple term
            test_vector = test_embeddings.embed_query("test")
            if len(test_vector) == 384:  # Expected dimension
                self.embedding_service_available = True
                print("✅ Ollama embeddings available with pankajrajdeo/biomed-embeddings-16l-fp16")
                return
        except Exception as e:
            print(f"⚠️ Ollama embeddings not available: {e}")
        
        # Fallback to external service if specified
        if self.embedding_service_url:
            try:
                response = requests.get(f"{self.embedding_service_url}/health", timeout=5)
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get('status') == 'healthy':
                        self.embedding_service_available = True
                        print(f"✅ External embedding service available at {self.embedding_service_url}")
                    else:
                        print(f"⚠️ External embedding service unhealthy at {self.embedding_service_url}")
                else:
                    print(f"❌ External embedding service not responding at {self.embedding_service_url}")
            except Exception as e:
                print(f"❌ Cannot connect to external embedding service: {e}")
        
        if not self.embedding_service_available:
            print("⚠️ No embedding service available - semantic search will be disabled")
            self.embedding_service_available = False

    def _ensure_schema_exists(self):
        """Ensure the PostgreSQL schema and functions exist. Fast startup version."""
        
        # FAST STARTUP: Only create essential table and extensions, skip expensive index operations
        schema_sql = """
        -- Enable extensions (fast)
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE EXTENSION IF NOT EXISTS pg_trgm;
        CREATE EXTENSION IF NOT EXISTS unaccent;

        -- Create table if not exists (fast)
        CREATE TABLE IF NOT EXISTS ontology_terms (
            id SERIAL PRIMARY KEY,
            ontology_id VARCHAR(50) NOT NULL,
            ontology VARCHAR(20) NOT NULL,
            label TEXT NOT NULL,
            definition TEXT,
            synonyms TEXT[],
            combined_text TEXT,
            embedding vector(384),
            combined_text_tsvector tsvector,
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(ontology, ontology_id)
        );

        -- Also ensure the tsvector column exists, it's fast to check and add.
        ALTER TABLE ontology_terms ADD COLUMN IF NOT EXISTS combined_text_tsvector tsvector;
        """
        
        # Check if essential indexes exist (very fast query)
        index_check_sql = """
        SELECT COUNT(*) as existing_indexes
        FROM pg_indexes 
        WHERE tablename = 'ontology_terms' 
        AND indexname IN ('idx_ontology_terms_ontology_id', 'idx_ontology_terms_ontology');
        """
        
        with self.conn.cursor() as cur:
            # Step 1: Fast schema setup (table + extensions only)
            cur.execute(schema_sql)
            
            # Step 2: Check if we need to create indexes and functions (fast check)
            cur.execute(index_check_sql)
            result = cur.fetchone()
            existing_indexes = result['existing_indexes'] if result else 0
            
            # Step 3: Only create indexes/functions if they don't exist (skip for speed)
            if existing_indexes < 2:
                print("⚠️  Essential indexes missing - consider running 'python ontology_manager.py rebuild-indexes' for optimal performance")
            
            # Step 4: Only ensure functions exist (much faster than recreating)
            cur.execute("SELECT COUNT(*) as function_count FROM pg_proc WHERE proname = 'exact_search'")
            result = cur.fetchone()
            function_exists = (result['function_count'] if result else 0) > 0
            
            if not function_exists:
                self._create_essential_functions(cur)
                
        self.conn.commit()
        print("✅ Database schema ready (fast startup mode)")
    
    def _create_essential_functions(self, cur):
        """Create only the essential database functions needed for operation."""
        functions_sql = """
        -- Drop functions to ensure signatures are updated
        DROP FUNCTION IF EXISTS exact_search(text, text, text);
        DROP FUNCTION IF EXISTS fulltext_search(text, text);
        DROP FUNCTION IF EXISTS fuzzy_search(text, text, real);
        DROP FUNCTION IF EXISTS semantic_search(vector(384), text, real);

        CREATE OR REPLACE FUNCTION exact_search(query_text TEXT, query_text_lower TEXT, ontology_filter TEXT DEFAULT NULL)
        RETURNS TABLE(
            ontology_id VARCHAR,
            ontology VARCHAR,
            label TEXT,
            definition TEXT,
            synonyms TEXT[]
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT t.ontology_id, t.ontology, t.label, t.definition, t.synonyms
            FROM ontology_terms t
            WHERE 
            (
                t.label = query_text OR -- Case-sensitive match on standard index
                LOWER(t.label) = query_text_lower OR -- Case-insensitive match on functional index
                t.ontology_id = query_text OR
                LOWER(t.ontology_id) = query_text_lower OR
                t.synonyms @> ARRAY[query_text] OR -- Case-sensitive synonym match on GIN index
                t.synonyms @> ARRAY[query_text_lower] -- Case-insensitive synonym match on GIN index
            )
            AND (ontology_filter IS NULL OR t.ontology = ANY(string_to_array(ontology_filter, ',')))
            LIMIT 10;
        END;
        $$ LANGUAGE plpgsql;

        CREATE OR REPLACE FUNCTION fulltext_search(query_text TEXT, ontology_filter TEXT DEFAULT NULL)
        RETURNS TABLE(
            ontology_id VARCHAR,
            ontology VARCHAR,
            label TEXT,
            definition TEXT,
            synonyms TEXT[],
            rank REAL
        ) AS $$
        DECLARE
            ts_query_plain tsquery := plainto_tsquery('english', query_text);
            ts_query_phrase tsquery := phraseto_tsquery('english', query_text);
        BEGIN
            RETURN QUERY
            SELECT t.ontology_id, t.ontology, t.label, t.definition, t.synonyms,
                   (ts_rank(t.combined_text_tsvector, ts_query_plain) + ts_rank(t.combined_text_tsvector, ts_query_phrase))::REAL as rank
            FROM ontology_terms t
            WHERE t.combined_text_tsvector @@ ts_query_plain
            AND (ontology_filter IS NULL OR t.ontology = ANY(string_to_array(ontology_filter, ',')))
            ORDER BY rank DESC
            LIMIT 10;
        END;
        $$ LANGUAGE plpgsql;
        
        CREATE OR REPLACE FUNCTION fuzzy_search(query_text TEXT, ontology_filter TEXT DEFAULT NULL, threshold REAL DEFAULT 0.3)
        RETURNS TABLE(
            ontology_id VARCHAR,
            ontology VARCHAR,
            label TEXT,
            definition TEXT,
            synonyms TEXT[],
            similarity REAL
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT t.ontology_id, t.ontology, t.label, t.definition, t.synonyms,
                   similarity(t.label, query_text) AS similarity
            FROM ontology_terms t
            WHERE t.label % query_text -- Use the % operator for trigram index
            AND similarity(t.label, query_text) >= threshold
            AND (ontology_filter IS NULL OR t.ontology = ANY(string_to_array(ontology_filter, ',')))
            ORDER BY similarity DESC
            LIMIT 10;
        END;
        $$ LANGUAGE plpgsql;

        CREATE OR REPLACE FUNCTION semantic_search(query_embedding VECTOR(384), ontology_filter TEXT DEFAULT NULL, threshold REAL DEFAULT 0.5)
        RETURNS TABLE(
            ontology_id VARCHAR,
            ontology VARCHAR,
            label TEXT,
            definition TEXT,
            synonyms TEXT[],
            similarity REAL
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT t.ontology_id, t.ontology, t.label, t.definition, t.synonyms,
                   (1 - (t.embedding <=> query_embedding))::REAL AS similarity
            FROM ontology_terms t
            WHERE (1 - (t.embedding <=> query_embedding)) >= threshold
            AND (ontology_filter IS NULL OR t.ontology = ANY(string_to_array(ontology_filter, ',')))
            ORDER BY similarity DESC
            LIMIT 5;
        END;
        $$ LANGUAGE plpgsql;
        """
        cur.execute(functions_sql)

    def _get_embedding_from_service(self, text: str) -> Optional[List[float]]:
        """Get a single embedding using sliding window approach for long texts."""
        return self.generate_sliding_window_embedding(text)

    def _get_batch_embeddings_from_service(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Get batch embeddings using sliding window approach for long texts."""
        if not self.embedding_service_available:
            return [None] * len(texts)
        
        try:
            # Separate short and long texts for efficient processing
            short_texts = []
            long_texts = []
            text_indices = []  # Track original positions
            
            for i, text in enumerate(texts):
                if len(text) <= self.max_text_length:
                    short_texts.append(text)
                    text_indices.append(('short', len(short_texts) - 1, i))
                else:
                    long_texts.append(text)
                    text_indices.append(('long', len(long_texts) - 1, i))
            
            print(f"   📊 Processing {len(short_texts)} short texts, {len(long_texts)} long texts (sliding window)")
            
            # Process short texts in batch (fast)
            short_embeddings = []
            if short_texts:
                short_embeddings = self._get_batch_embeddings_direct(short_texts)
            
            # Process long texts with sliding window (slower but better quality)
            long_embeddings = []
            if long_texts:
                for i, long_text in enumerate(long_texts):
                    print(f"   🔀 Processing long text {i+1}/{len(long_texts)}")
                    embedding = self.generate_sliding_window_embedding(long_text)
                    long_embeddings.append(embedding)
            
            # Reconstruct embeddings in original order
            all_embeddings = [None] * len(texts)
            for text_type, type_index, original_index in text_indices:
                if text_type == 'short':
                    all_embeddings[original_index] = short_embeddings[type_index] if type_index < len(short_embeddings) else None
                else:  # long
                    all_embeddings[original_index] = long_embeddings[type_index] if type_index < len(long_embeddings) else None
            
            return all_embeddings
            
        except Exception as e:
            print(f"Error in batch embedding processing: {e}")
            return [None] * len(texts)

    def clean_text(self, text: str) -> str:
        """Clean and normalize text for processing."""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Remove HTML-like tags
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        
        # Remove extra quotes
        cleaned = cleaned.replace('"', '').replace("'", "")
        
        return cleaned
    
    def generate_sliding_window_embedding(self, text: str, window_size: int = 1450, overlap: int = 50) -> Optional[List[float]]:
        """Generate embedding using sliding window approach for long texts with averaging."""
        if not text:
            return None
            
        # For short texts, use standard single embedding
        if len(text) <= self.max_text_length:
            return self._get_single_embedding_direct(text)
        
        # Create overlapping chunks for long texts
        chunks = []
        step = window_size - overlap
        
        for i in range(0, len(text), step):
            chunk = text[i:i + window_size]
            if len(chunk.strip()) > 50:  # Skip tiny meaningless chunks
                chunks.append(chunk)
        
        if not chunks:
            return None
            
        print(f"   🔀 Splitting {len(text)} chars into {len(chunks)} overlapping windows")
        
        # Get embeddings for all chunks
        chunk_embeddings = self._get_batch_embeddings_direct(chunks)
        valid_embeddings = [emb for emb in chunk_embeddings if emb is not None]
        
        if not valid_embeddings:
            return None
        
        # Average the embeddings for final representation
        try:
            # Convert to numpy array and handle any None values
            valid_embeddings_array = np.array([emb for emb in valid_embeddings if emb is not None and len(emb) == 384])
            
            if len(valid_embeddings_array) == 0:
                print("   ⚠️  No valid embeddings to average")
                return None
            
            # Check for NaN or infinite values
            if np.any(np.isnan(valid_embeddings_array)) or np.any(np.isinf(valid_embeddings_array)):
                print("   ⚠️  Invalid embeddings detected (NaN/Inf), using first valid embedding")
                return valid_embeddings[0]
            
            averaged_embedding = np.mean(valid_embeddings_array, axis=0).tolist()
            
            # Final validation
            if any(np.isnan(x) or np.isinf(x) for x in averaged_embedding):
                print("   ⚠️  Averaged embedding contains invalid values, using first valid embedding")
                return valid_embeddings[0]
                
            print(f"   📊 Averaged {len(valid_embeddings_array)}/{len(chunks)} valid chunk embeddings")
            return averaged_embedding
            
        except Exception as e:
            print(f"   ⚠️  Error during embedding averaging: {e}, using first valid embedding")
            return valid_embeddings[0] if valid_embeddings else None
    
    def _get_single_embedding_direct(self, text: str) -> Optional[List[float]]:
        """Get a single embedding using Ollama."""
        if not self.embedding_service_available:
            return None
        
        try:
            from langchain_ollama import OllamaEmbeddings
            embeddings = OllamaEmbeddings(model="pankajrajdeo/biomed-embeddings-16l-fp16")
            embedding = embeddings.embed_query(text)
            
            # Validate embedding
            if embedding and len(embedding) == 384 and not any(np.isnan(x) or np.isinf(x) for x in embedding):
                return embedding
            else:
                print(f"⚠️  Invalid embedding generated (length: {len(embedding) if embedding else 0})")
                return None
                
        except Exception as e:
            print(f"⚠️  Error calling Ollama embedding service: {e}")
            return None
    
    def _get_batch_embeddings_direct(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Get batch embeddings using Ollama."""
        if not self.embedding_service_available:
            return [None] * len(texts)
        
        try:
            from langchain_ollama import OllamaEmbeddings
            embeddings = OllamaEmbeddings(model="pankajrajdeo/biomed-embeddings-16l-fp16")
            
            all_embeddings = []
            for text in texts:
                try:
                    embedding = embeddings.embed_query(text)
                    # Validate embedding
                    if embedding and len(embedding) == 384 and not any(np.isnan(x) or np.isinf(x) for x in embedding):
                        all_embeddings.append(embedding)
                    else:
                        print(f"⚠️  Invalid embedding generated for text (length: {len(embedding) if embedding else 0})")
                        all_embeddings.append(None)
                except Exception as e:
                    print(f"⚠️  Error generating embedding for text: {e}")
                    all_embeddings.append(None)
            
            return all_embeddings
            
        except Exception as e:
            print(f"Error calling Ollama embedding service: {e}")
            return [None] * len(texts)
    
    def extract_ontology_id(self, uri: str, ontology: str) -> Optional[str]:
        """Extract the ontology ID from a URI using the configured pattern."""
        if not uri:
            return None
        
        config = self.ontology_configs.get(ontology)
        if not config:
            return None
        
        pattern = config['id_pattern']
        match = re.search(pattern, uri)
        if match:
            return match.group(1)
        
        return None
    
    def extract_synonyms(self, cls, namespaces: Dict[str, str]) -> List[str]:
        """Extract synonyms from an OWL class element."""
        synonyms = []
        
        # Look for various synonym properties
        synonym_properties = [
            './/oboInOwl:hasExactSynonym',
            './/oboInOwl:hasBroadSynonym', 
            './/oboInOwl:hasNarrowSynonym',
            './/oboInOwl:hasRelatedSynonym',
            './/skos:altLabel',
            './/rdfs:label'
        ]
        
        for prop in synonym_properties:
            for elem in cls.findall(prop, namespaces):
                if elem.text:
                    synonym = self.clean_text(elem.text)
                    if synonym and synonym not in synonyms:
                        synonyms.append(synonym)
        
        return synonyms
    
    def extract_definition(self, cls, namespaces: Dict[str, str]) -> str:
        """Extract definition/description from an OWL class."""
        definition_properties = [
            './/obo:IAO_0000115',  # OBO definition
            './/rdfs:comment',     # RDFS comment
            './/skos:definition',  # SKOS definition
            './/dc:description'    # Dublin Core description
        ]
        
        for prop in definition_properties:
            for elem in cls.findall(prop, namespaces):
                if elem.text:
                    definition = self.clean_text(elem.text)
                    if definition and len(definition) > 10:  # Meaningful definition
                        return definition
        
        return ""
    
    def parse_owl_file(self, file_path: str, ontology: str) -> List[Dict]:
        """Parse an OWL file and extract ontology terms."""
        if not os.path.exists(file_path):
            print(f"Warning: OWL file not found: {file_path}")
            return []
        
        print(f"Parsing {ontology} from {file_path}...")
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return []
            
        terms = []
        classes = root.findall('.//owl:Class', self.namespaces)
        
        for cls in tqdm(classes, desc=f"Processing {ontology} terms"):
            # Extract ontology ID from rdf:about
            about = cls.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
            if not about:
                continue
                
            ontology_id = self.extract_ontology_id(about, ontology)
            if not ontology_id:
                continue
                
            # Extract label
            label_elem = cls.find('.//rdfs:label', self.namespaces)
            if label_elem is None or not label_elem.text:
                continue
                
            label = self.clean_text(label_elem.text)
            if not label:
                continue
                
            # Extract synonyms
            synonyms = self.extract_synonyms(cls, self.namespaces)
                
            # Extract definition
            definition = self.extract_definition(cls, self.namespaces)
            
            # Combine text for full-text search
            combined_parts = [label]
            if definition:
                combined_parts.append(definition)
            combined_parts.extend(synonyms)
            combined_text = ' '.join(combined_parts)
                
            term = {
                'ontology_id': ontology_id,
                'ontology': ontology,
                'label': label,
                'definition': definition,
                'synonyms': synonyms,
                'combined_text': combined_text,
                'embedding': None  # Will be computed later
            }
            
            terms.append(term)
        
        print(f"Extracted {len(terms)} terms from {ontology}")
        return terms
            
    def load_ontology_to_database(self, ontology: str, force_reload: bool = False):
        """Load a specific ontology directly into PostgreSQL."""
        
        # Check if ontology already exists in database
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as count FROM ontology_terms WHERE ontology = %s", (ontology,))
            existing_count = cur.fetchone()['count']
        
        if existing_count > 0 and not force_reload:
            print(f"Ontology {ontology} already loaded ({existing_count} terms). Use force_reload=True to reload.")
            return
        
        # Parse OWL file
        config = self.ontology_configs.get(ontology)
        if not config:
            print(f"Unknown ontology: {ontology}")
            return
        
        terms = self.parse_owl_file(config['file'], ontology)
        if not terms:
            print(f"No terms extracted from {ontology}")
            return
        
        # Compute embeddings if service is available
        if self.embedding_service_available:
            print(f"Computing embeddings for {len(terms)} {ontology} terms...")
            texts = [term['combined_text'] for term in terms]
            embeddings = self._get_batch_embeddings_from_service(texts)
            
            for term, embedding in zip(terms, embeddings):
                if embedding:
                    term['embedding'] = f"[{','.join(map(str, embedding))}]"
        
        # Always clear old data for a clean load
        with self.conn.cursor() as cur:
            print(f"Clearing old data for {ontology}...")
            cur.execute("DELETE FROM ontology_terms WHERE ontology = %s", (ontology,))
        self.conn.commit()

        # Parse OWL file
        print(f"🔍 Parsing {ontology} from {config['file']}...")
        terms = self.parse_owl_file(config['file'], ontology)
        
        # Prepare data for insertion
        print(f"Preparing {len(terms)} terms for insertion...")
        insert_data = []
        for term in terms:
            label = term.get('label', '')
            definition = term.get('definition', '')
            synonyms = term.get('synonyms', [])
            combined_text = f"{label} {' '.join(synonyms)} {definition}"
            
            insert_data.append((
                term['id'],
                ontology,
                label,
                definition,
                synonyms,
                combined_text,
                combined_text # Pass it again for the tsvector function
            ))
        
        # Use execute_values for efficient bulk insertion
        print(f" Inserting {len(insert_data):,} records...")
        with self.conn.cursor() as cur:
            # The template automatically calls the to_tsvector function on the last %s
            template = '(%s, %s, %s, %s, %s, %s, to_tsvector(\'english\', %s))'
            execute_values(
                cur,
                f"""
                INSERT INTO ontology_terms (ontology_id, ontology, label, definition, synonyms, combined_text, combined_text_tsvector)
                VALUES %s
                """,
                insert_data,
                template=template
            )

        self.conn.commit()
        print(f"✅ Successfully loaded {len(terms)} {ontology} terms")
        
        # We only create vector index if embeddings are available and needed
        if self.embedding_service_available:
            self._create_vector_index()

    def _create_vector_index(self):
        """Create vector index if it doesn't exist and we have embedding data."""
        try:
            with self.conn.cursor() as cur:
                # First check if we have embeddings before creating the index
                cur.execute("SELECT COUNT(*) as count FROM ontology_terms WHERE embedding IS NOT NULL")
                result = cur.fetchone()
                embedding_count = result['count'] if result else 0
                
                if embedding_count == 0:
                    print("⚠️  No embeddings found, skipping vector index creation")
                    return
                
                # Use fewer lists for lower memory usage (100 lists ~= 27MB instead of 273MB)
                # This is still effective for performance with good clustering
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ontology_terms_embedding 
                    ON ontology_terms USING ivfflat (embedding vector_cosine_ops) 
                    WITH (lists = 100)
                """)
            self.conn.commit()
            print(f"✅ Vector index created/updated for {embedding_count:,} embeddings")
        except Exception as e:
            # We need to rollback if index creation fails inside its transaction
            self.conn.rollback()
            print(f"Note: Vector index creation skipped: {e}")

    def load_all_ontologies(self, force_reload: bool = False):
        """Load all configured ontologies into PostgreSQL."""
        print("Loading all ontologies into PostgreSQL...")
        
        for ontology in self.ontology_configs.keys():
            try:
                self.load_ontology_to_database(ontology, force_reload=force_reload)
            except Exception as e:
                print(f"Error loading {ontology}: {e}")
                continue
                
    def _process_ontology_filter(self, ontology_filter: Optional[str]) -> Optional[str]:
        """
        Process ontology filter to handle comma-separated lists and spaces.
        Returns a cleaned comma-separated string or None.
        """
        if not ontology_filter or ontology_filter.strip() == '':
            return None
        
        # Split by comma, trim whitespace, and remove empty strings
        ontologies = [ont.strip() for ont in ontology_filter.split(',') if ont.strip()]
        
        if not ontologies:
            return None
            
        return ','.join(ontologies)

    def exact_match(self, query: str, ontology_filter: Optional[str] = None) -> List[Dict]:
        """Search for exact matches."""
        processed_filter = self._process_ontology_filter(ontology_filter)
        with self.conn.cursor() as cur:
            # Pass both original and lowercased query to the SQL function for optimal index usage.
            cur.execute("SELECT * FROM exact_search(%s, %s, %s)", (query, query.lower(), processed_filter))
            return [dict(row) for row in cur.fetchall()]

    def fulltext_match(self, query: str, ontology_filter: Optional[str] = None) -> List[Dict]:
        """Search using full-text search."""
        processed_filter = self._process_ontology_filter(ontology_filter)
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM fulltext_search(%s, %s)", (query, processed_filter))
            return [dict(row) for row in cur.fetchall()]

    def fuzzy_match(self, query: str, ontology_filter: Optional[str] = None, threshold: float = 0.3) -> List[Dict]:
        """Search using fuzzy string matching."""
        processed_filter = self._process_ontology_filter(ontology_filter)
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM fuzzy_search(%s, %s, %s)", (query, processed_filter, threshold))
            return [dict(row) for row in cur.fetchall()]

    def embedding_match(self, query: str, ontology_filter: Optional[str] = None, threshold: float = 0.5) -> List[Dict]:
        """Search using semantic similarity with embeddings."""
        processed_filter = self._process_ontology_filter(ontology_filter)
        if not self.embedding_service_available:
            print("⚠️  Embedding service not available, skipping semantic search")
            return []
        
        embedding = self._get_embedding_from_service(query)
        if not embedding:
            print("⚠️  Could not generate embedding for query")
            return []
        
        embedding_str = f"[{','.join(map(str, embedding))}]"
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM semantic_search(%s, %s, %s)", 
                       (embedding_str, processed_filter, threshold))
            return [dict(row) for row in cur.fetchall()]
    
    def search(self, query: str, ontology_filter: Optional[str] = None) -> Dict:
        """Multi-tier search using all PostgreSQL-optimized methods."""
        start_time = time.time()
        
        results = {
            'query': query,
            'ontology_filter': ontology_filter,
            'exact_matches': self.exact_match(query, ontology_filter),
            'fulltext_matches': self.fulltext_match(query, ontology_filter),
            'fuzzy_matches': self.fuzzy_match(query, ontology_filter),
            'embedding_matches': self.embedding_match(query, ontology_filter)
        }
        
        search_time = time.time() - start_time
        results['search_time'] = search_time
        
        print(f"🔍 Search completed in {search_time:.3f}s")
        return results
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        with self.conn.cursor() as cur:
            # Total terms
            cur.execute("SELECT COUNT(*) as total FROM ontology_terms")
            total_terms = cur.fetchone()['total']
            
            # Terms with embeddings
            cur.execute("SELECT COUNT(*) as with_embeddings FROM ontology_terms WHERE embedding IS NOT NULL")
            terms_with_embeddings = cur.fetchone()['with_embeddings']
            
            # Terms per ontology
            cur.execute("SELECT ontology, COUNT(*) as count FROM ontology_terms GROUP BY ontology ORDER BY ontology")
            ontology_counts = cur.fetchall()
            
            # Database size
            cur.execute("SELECT pg_size_pretty(pg_database_size(current_database())) as db_size")
            db_size = cur.fetchone()['db_size']
            
            return {
                'total_terms': total_terms,
                'terms_with_embeddings': terms_with_embeddings,
                'ontology_counts': {row['ontology']: row['count'] for row in ontology_counts},
                'database_size': db_size
            }

    def get_term_by_id(self, ontology_id: str, ontology: Optional[str] = None) -> Optional[Dict]:
        """Get a specific term by its ontology ID."""
        with self.conn.cursor() as cur:
            if ontology:
                cur.execute("SELECT * FROM ontology_terms WHERE ontology_id = %s AND ontology = %s", 
                           (ontology_id, ontology))
            else:
                cur.execute("SELECT * FROM ontology_terms WHERE ontology_id = %s", (ontology_id,))
            result = cur.fetchone()
            return dict(result) if result else None

    def list_ontologies(self) -> List[str]:
        """List all available ontologies."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT DISTINCT ontology FROM ontology_terms ORDER BY ontology")
            results = cur.fetchall()
            return [row['ontology'] for row in results]
    
    def save_to_json(self, output_file: str = "unified_ontology_mapping.json"):
        """Export database contents to JSON format for backup."""
        print(f"Exporting database to {output_file}...")
        
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM ontology_terms ORDER BY ontology, ontology_id")
            all_terms = cur.fetchall()
        
        # Convert to JSON-serializable format
        terms_list = []
        for row in all_terms:
            term = dict(row)
            # Convert embedding vector to list if it exists
            if term.get('embedding'):
                # PostgreSQL vector comes as string, convert to list
                embedding_str = str(term['embedding'])
                if embedding_str.startswith('[') and embedding_str.endswith(']'):
                    try:
                        term['embedding'] = [float(x) for x in embedding_str[1:-1].split(',')]
                    except:
                        term['embedding'] = None
            terms_list.append(term)
        
        # Create output structure
        output_data = {
            'metadata': {
                'total_terms': len(terms_list),
                'ontologies': list(set(term['ontology'] for term in terms_list)),
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'description': 'Unified ontology mapping exported from PostgreSQL'
            },
            'terms': terms_list
        }
        
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Exported {len(terms_list)} terms to {output_file}")
        
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def rebuild_all_indexes(self):
        """
        Unconditionally drops and recreates all performance-critical indexes.
        This is an expensive operation intended for maintenance.
        Manages its own transaction and autocommit state carefully.
        """
        # First, commit any lingering transaction from the connection setup.
        # This is crucial before changing the autocommit session variable.
        self.conn.commit()
        
        original_autocommit = self.conn.autocommit
        try:
            # Switch to autocommit mode for index creation and vacuuming
            self.conn.autocommit = True
            with self.conn.cursor() as cur:
                print("  -> Dropping old indexes...")
                cur.execute("""
                    DROP INDEX IF EXISTS idx_ontology_terms_ontology_id;
                    DROP INDEX IF EXISTS idx_ontology_terms_ontology;
                    DROP INDEX IF EXISTS idx_ontology_terms_label;
                    DROP INDEX IF EXISTS idx_ontology_terms_label_gin;
                    DROP INDEX IF EXISTS idx_ontology_terms_definition_gin;
                    DROP INDEX IF EXISTS idx_ontology_terms_synonyms_gin;
                    DROP INDEX IF EXISTS idx_ontology_terms_trigram;
                    DROP INDEX IF EXISTS idx_ontology_terms_combined_gin;
                    DROP INDEX IF EXISTS idx_ontology_terms_label_lower;
                    DROP INDEX IF EXISTS idx_ontology_terms_ontology_id_lower;
                    DROP INDEX IF EXISTS idx_ontology_terms_embedding;
                """)
                
                print("  -> Creating new performance-critical indexes...")
                cur.execute("CREATE INDEX idx_ontology_terms_ontology_id ON ontology_terms(ontology_id)")
                cur.execute("CREATE INDEX idx_ontology_terms_ontology ON ontology_terms(ontology)")
                cur.execute("CREATE INDEX idx_ontology_terms_label_lower ON ontology_terms(LOWER(label))")
                cur.execute("CREATE INDEX idx_ontology_terms_ontology_id_lower ON ontology_terms(LOWER(ontology_id))")
                cur.execute("CREATE INDEX idx_ontology_terms_synonyms_gin ON ontology_terms USING GIN(synonyms)")
                cur.execute("CREATE INDEX idx_ontology_terms_trigram ON ontology_terms USING GIN(label gin_trgm_ops)")
                cur.execute("CREATE INDEX idx_ontology_terms_combined_gin ON ontology_terms USING GIN(combined_text_tsvector)")

                print("  -> Running VACUUM ANALYZE for optimal query planning...")
                cur.execute("VACUUM ANALYZE ontology_terms")

        finally:
            # Always restore the original autocommit state
            self.conn.autocommit = original_autocommit

        # The vector index creation manages its own transaction, so it's called outside the autocommit block.
        print("  -> Creating vector index (if embeddings exist)...")
        self._create_vector_index()
        
        print("  -> Creating/updating search functions...")
        with self.conn.cursor() as cur:
            self._create_essential_functions(cur)
        self.conn.commit()

        print("✅ All indexes rebuilt and database vacuumed.")


def main():
    """Main function to run the unified ontology mapper."""
    parser = argparse.ArgumentParser(description='Unified Ontology Mapper with PostgreSQL Backend')
    parser.add_argument('--load-ontologies', action='store_true', 
                       help='Load all OWL files into PostgreSQL')
    parser.add_argument('--load-ontology', type=str, 
                       help='Load specific ontology (e.g., PATO, UBERON)')
    parser.add_argument('--force-reload', action='store_true',
                       help='Force reload even if ontology exists in database')
    parser.add_argument('--search', type=str,
                       help='Search for a term')
    parser.add_argument('--ontology-filter', type=str,
                       help='Filter search to specific ontology')
    parser.add_argument('--export-json', type=str,
                       help='Export database to JSON file')
    parser.add_argument('--stats', action='store_true',
                       help='Show database statistics')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive search session')
    
    args = parser.parse_args()
    
    try:
        with UnifiedOntologyMapper() as mapper:
            
            if args.load_ontologies:
                mapper.load_all_ontologies(force_reload=args.force_reload)
            
            elif args.load_ontology:
                ontology = args.load_ontology.upper()
                if ontology in mapper.ontology_configs:
                    mapper.load_ontology_to_database(ontology, force_reload=args.force_reload)
                else:
                    print(f"Unknown ontology: {ontology}")
                    print(f"Available: {', '.join(mapper.ontology_configs.keys())}")
            
            elif args.search:
                results = mapper.search(args.search, args.ontology_filter)
                # Pretty print results
                import pprint
                pprint.pprint(results)
            
            elif args.export_json:
                mapper.save_to_json(args.export_json)
                
            elif args.stats:
                import pprint
                pprint.pprint(mapper.get_stats())
            
            elif args.interactive:
                print("Starting interactive search session. Type 'exit' to quit.")
                while True:
                    query = input("Enter search term> ")
                    if query.lower() == 'exit':
                        break
                    results = mapper.search(query)
                    import pprint
                    pprint.pprint(results)
                    
    except psycopg2.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main() 