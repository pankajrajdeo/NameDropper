#!/usr/bin/env python3
"""
Metadata Harmonization Script with a Sequential LLM Workflow.

This script implements a two-step "Search then Analyze" workflow:
1. It acts as a client to the PostgreSQL-based UnifiedOntologyMapper to SEARCH for
    potential ontology matches for a given term.
2. It then feeds these search results into a second LLM call (the "Analyst")
    to ANALYZE the candidates and select the single best match based on
    context and a set of explicit rules.
"""

import json
import os
from typing import Dict, Any, Optional, List
from tqdm import tqdm
from datetime import datetime
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

# Import from the same package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.ontology_mapper import UnifiedOntologyMapper

# --- Pydantic Models for Gradio App Interface ---
class OntologyMatch(BaseModel):
    """Represents a single ontology match."""
    ontology_id: str
    label: str
    ontology: str
    definition: Optional[str] = None
    similarity: Optional[float] = None

class HarmonizedTerm(BaseModel):
    """Represents the result of harmonizing a single term."""
    status: str  # "success", "no_match", "error"
    method: str  # Which search method found the match
    confidence: float  # 0.0 to 1.0
    best_match: Optional[OntologyMatch] = None
    alternative_matches: List[OntologyMatch] = []
    reasoning: str
    search_summary: Optional[Dict[str, int]] = None  # Summary of search results

# --- Configuration for the Analyst LLM ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANALYSIS_MODEL = "gpt-4o-mini"  # OpenAI's latest model

# Initialize ChatOpenAI instance (lazy initialization)
llm = None

def get_llm():
    """Get LLM instance, initializing if needed."""
    global llm
    if llm is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set")
        llm = ChatOpenAI(
            model=ANALYSIS_MODEL,
            temperature=0.1,
            max_tokens=1024,
            timeout=60,
            max_retries=2,
            api_key=OPENAI_API_KEY
        )
    return llm

# --- The Universal Prompt Template for the Analyst LLM ---
ANALYSIS_PROMPT_TEMPLATE = """
You are an ontology harmonization expert. Analyze the search results and select the single best match for the given term.

**INPUT TERM**: "{original_term}"
**FIELD**: {field_type}
**DOMAIN**: {domain_context}
**DESCRIPTION**: {dataset_description}

**DECISION RULES** (apply in order):
1. EXACT MATCH → Select immediately (confidence: 1.0)
2. FUZZY MATCH → Use if similarity ≥ 0.95  
3. SEMANTIC MATCH → Use if similarity ≥ 0.85 AND contextually relevant
4. FULLTEXT MATCH → Use if relevant to domain
5. NO MATCH → Return null if none apply

**SEARCH RESULTS**:
{search_results_json}

**DOMAIN GUIDANCE**: {domain_guidance}

**OUTPUT** (JSON only):
{{
  "harmonized_term": "selected label or null",
  "ontology_id": "selected ID or null", 
  "ontology_source": "ontology code or null",
  "confidence_score": 0.0-1.0,
  "reasoning": "Rule X: brief explanation"
}}
"""

def expand_query_with_llm(term: str, field: str, description: str, n: int = 5) -> List[str]:
    """
    Uses a context-aware LLM to expand a search term into a list of variations.
    """
    print(f"   [AGENT] Expanding query: '{term}' for field '{field}' (n={n}) with context...")
    
    if not OPENAI_API_KEY:
        print("   [AGENT] ❌ OPENAI_API_KEY not found. Skipping query expansion.")
        return [term]

    prompt = f"""
You are a biomedical ontology expert. Your task is to expand a given search term into a list of likely variations for a database search, guided by the provided context.
Generate a JSON object with a single key "queries" which contains a list of strings.

**Strictly adhere to these rules:**
1. Use the **Dataset Description** to understand the context.
2. Generate **exactly {n}** variations relevant to the context, unless fewer are possible.
3. Focus on orthographic variations (e.g., AT1, AT-1), common abbreviations, and official full names that fit the context.
4. **DO NOT** invent new terms. Stick to plausible, near-similar terms. Preserve casing.

**INPUT TERM**: "{term}"
**FIELD CONTEXT**: "{field}"
**DATASET DESCRIPTION**: "{description}"

**OUTPUT (JSON object with a "queries" key only!)**:
"""
    
    try:
        messages = [("system", "You must respond with valid JSON only."), ("user", prompt)]
        response = get_llm().invoke(messages)
        llm_output_str = response.content
        
        try:
            expanded_data = json.loads(llm_output_str)
            expanded_list = expanded_data.get("queries", [])

            if isinstance(expanded_list, list) and all(isinstance(s, str) for s in expanded_list):
                if term not in expanded_list:
                    expanded_list.insert(0, term)
                
                unique_list = list(dict.fromkeys(expanded_list))
                final_list = unique_list[:n]
                
                print(f"   [AGENT] ✅ Expanded to: {final_list} (truncated to n={n})")
                return final_list
            else:
                print(f"   [AGENT] ⚠️ LLM returned unexpected format for query expansion. Using original term.")
                return [term]
        except json.JSONDecodeError:
            print(f"   [AGENT] ⚠️ LLM returned invalid JSON. Using original term.")
            return [term]

    except Exception as e:
        print(f"   [AGENT] 🔥 Error during query expansion: {e}. Using original term.")
        return [term]

def harmonize_metadata(
    dataset_description: str,
    term_name: str,
    field_name: str,
    mapper: UnifiedOntologyMapper,
    ontologies: Optional[str] = None,
    k: int = 5,
    use_query_expansion: bool = False,
    num_expanded_queries: int = 5
) -> HarmonizedTerm:
    """
    Harmonize a single metadata term, with optional query expansion.
    """
    try:
        # STEP 1: EXPAND (optional)
        if use_query_expansion:
            if not dataset_description.strip():
                print("   [AGENT] ⚠️ Query expansion skipped: Dataset Description is required.")
                expanded_queries = [term_name]
            else:
                expanded_queries = expand_query_with_llm(term_name, field_name, dataset_description, n=num_expanded_queries)
        else:
            expanded_queries = [term_name]

        # STEP 2: SEARCH
        print(f"   [AGENT] Searching with {len(expanded_queries)} query variations: {expanded_queries}")
        aggregated_results = {
            "exact_matches": [],
            "fuzzy_matches": [],
            "embedding_matches": [],
            "fulltext_matches": []
        }
        seen_ids = {"exact": set(), "fuzzy": set(), "embedding": set(), "fulltext": set()}

        for query in expanded_queries:
            search_results = mapper.search(query=query, ontology_filter=ontologies)
            
            # Merge results, avoiding duplicates
            for match in search_results.get("exact_matches", []):
                if match['ontology_id'] not in seen_ids['exact']:
                    aggregated_results['exact_matches'].append(match)
                    seen_ids['exact'].add(match['ontology_id'])
            
            for match in search_results.get("fuzzy_matches", []):
                if match['ontology_id'] not in seen_ids['fuzzy']:
                    aggregated_results['fuzzy_matches'].append(match)
                    seen_ids['fuzzy'].add(match['ontology_id'])

            for match in search_results.get("embedding_matches", []):
                if match['ontology_id'] not in seen_ids['embedding']:
                    aggregated_results['embedding_matches'].append(match)
                    seen_ids['embedding'].add(match['ontology_id'])

            for match in search_results.get("fulltext_matches", []):
                if match['ontology_id'] not in seen_ids['fulltext']:
                    aggregated_results['fulltext_matches'].append(match)
                    seen_ids['fulltext'].add(match['ontology_id'])
        
        # Add original query info for context
        aggregated_results['query'] = term_name
        aggregated_results['ontology_filter'] = ontologies
        
        # STEP 2.5: RE-SORT BY SIMILARITY
        if aggregated_results['fuzzy_matches']:
            aggregated_results['fuzzy_matches'].sort(key=lambda x: x.get('similarity', 0.0), reverse=True)
        if aggregated_results['embedding_matches']:
            aggregated_results['embedding_matches'].sort(key=lambda x: x.get('similarity', 0.0), reverse=True)

        # Create search summary
        search_summary = {
            "exact_matches": len(aggregated_results["exact_matches"]),
            "fuzzy_matches": len(aggregated_results["fuzzy_matches"]),
            "embedding_matches": len(aggregated_results["embedding_matches"]),
            "fulltext_matches": len(aggregated_results["fulltext_matches"])
        }
        
        # Check if we have any candidates
        has_candidates = any(search_summary.values())
        
        if not has_candidates:
            return HarmonizedTerm(
                status="no_match",
                method="query_expansion",
                confidence=0.0,
                reasoning=f"No candidates found by any search method for expanded queries: {expanded_queries}",
                search_summary=search_summary
            )
            
        # STEP 3: ANALYZE - Use LLM to select best match from aggregated results
        context = {
            "dataset_description": dataset_description,
            "original_term": term_name,
            "field_name": field_name,
            "domain_context": _get_domain_context(ontologies, field_name),
            "field_type": field_name,
            "domain_guidance": _get_domain_guidance(ontologies, field_name)
        }
        
        llm_result = get_llm_analysis(aggregated_results, context, k)
        
        if llm_result.get("ontology_id"):
            best_match = _find_match_details(llm_result, aggregated_results)
            alternatives = _get_alternative_matches(aggregated_results, llm_result, k)
            
            return HarmonizedTerm(
                status="success",
                method=_determine_method(llm_result, aggregated_results),
                confidence=llm_result.get("confidence_score", 0.0),
                best_match=best_match,
                alternative_matches=alternatives,
                reasoning=llm_result.get("reasoning", "No reasoning provided."),
                search_summary=search_summary
            )
        else:
            return HarmonizedTerm(
                status="no_match",
                method="llm_analysis",
                confidence=0.0,
                reasoning=llm_result.get("reasoning", "LLM Analyst did not select a match."),
                search_summary=search_summary
            )
    except Exception as e:
        print(f"🔥 UNEXPECTED ERROR in harmonize_metadata: {e}")
        return HarmonizedTerm(
            status="error",
            method="system",
            confidence=0.0,
            reasoning=f"An unexpected error occurred: {str(e)}",
            search_summary=None
        )

def _get_domain_context(ontologies: Optional[str], field_name: str) -> str:
    """Generate domain context based on ontologies and field name."""
    if ontologies:
        domain_mapping = {
            "HPO": "Clinical", "MONDO": "Clinical", "DOID": "Clinical", "ORDO": "Clinical",
            "MAXO": "Clinical", "OGMS": "Clinical",
            "CL": "Cellular", "UBERON": "Anatomical", "FMA": "Anatomical", 
            "RADLEX": "Radiological",
            "CHEBI": "Chemical", "GO": "Functional", "PR": "Molecular", "SO": "Genomic",
            "EFO": "Experimental", "OBI": "Methodological", "PATO": "Qualitative",
            "NCBITAXON": "Taxonomic", "HSAPDV": "Developmental", "MMUSDV": "Developmental",
            "GENO": "Genetic", "VO": "Immunological", "MPATH": "Pathological",
            "GSSO": "Demographic", "HANCESTRO": "Demographic"
        }
        
        ont_list = [ont.strip() for ont in ontologies.split(",")]
        domains = [domain_mapping.get(ont, "General") for ont in ont_list]
        return ", ".join(set(domains))
    return "Cross-domain"

def _get_domain_guidance(ontologies: Optional[str], field_name: str) -> str:
    """Generate domain-specific guidance."""
    field_hints = {
        "cell_type": "Prioritize CL (Cell Ontology) for cellular classifications, UBERON for tissue context",
        "disease": "Prioritize MONDO/DOID for diseases, HPO for phenotypes, ORDO for rare diseases", 
        "tissue": "Prioritize UBERON for general anatomy, FMA for detailed anatomical structures",
        "chemical": "Prioritize CHEBI for compounds, drugs, metabolites, and chemical entities",
        "anatomy": "Prioritize UBERON for general anatomy, FMA for detailed anatomical structures",
    }
    
    return field_hints.get(field_name.lower(), "Use standard ontology matching rules")

def get_llm_analysis(search_results: Dict, context: Dict, k: int) -> Dict[str, Any]:
    """Get LLM analysis of search results."""
    try:
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            original_term=context["original_term"],
            field_type=context["field_type"],
            domain_context=context["domain_context"],
            dataset_description=context["dataset_description"],
            search_results_json=json.dumps(search_results, indent=2),
            domain_guidance=context["domain_guidance"]
        )
        
        messages = [("user", prompt)]
        response = get_llm().invoke(messages)
        llm_output = response.content
        
        if llm_output:
            try:
                result = json.loads(llm_output)
                if not isinstance(result, dict):
                    raise ValueError("Response is not a JSON object")
                return result
            except json.JSONDecodeError as e:
                print(f"   [AGENT] 🔥 Invalid JSON in LLM response: {e}")
                return create_error_response("Invalid JSON in LLM response")
        
    except Exception as e:
        print(f"   [AGENT] 🔥 An unexpected error occurred in get_llm_analysis: {e}")
        return create_error_response(f"Unexpected error: {str(e)}")

def create_error_response(error_msg: str) -> Dict[str, Any]:
    """Create a standardized error response."""
    return {
        "harmonized_term": None,
        "ontology_id": None,
        "ontology_source": None,
        "confidence_score": 0.0,
        "reasoning": f"Analysis failed: {error_msg}"
    }

def _find_match_details(llm_result: Dict, search_results: Dict) -> Optional[OntologyMatch]:
    """Find the detailed match information from search results."""
    ontology_id = llm_result.get("ontology_id")
    if not ontology_id:
        return None
    
    # Search through all result types to find the match
    for result_type in ["exact_matches", "fuzzy_matches", "embedding_matches", "fulltext_matches"]:
        for match in search_results.get(result_type, []):
            if match.get("ontology_id") == ontology_id:
                return OntologyMatch(
                    ontology_id=match["ontology_id"],
                    label=match["label"],
                    ontology=match["ontology"],
                    definition=match.get("definition"),
                    similarity=match.get("similarity")
                )
    return None

def _get_alternative_matches(search_results: Dict, llm_result: Dict, k: int) -> List[OntologyMatch]:
    """Get alternative matches for the harmonized term."""
    alternatives = []
    ontology_id = llm_result.get("ontology_id")
    
    for result_type in ["exact_matches", "fuzzy_matches", "embedding_matches", "fulltext_matches"]:
        for match in search_results.get(result_type, []):
            if match.get("ontology_id") != ontology_id:
                alternatives.append(OntologyMatch(
                    ontology_id=match["ontology_id"],
                    label=match["label"],
                    ontology=match["ontology"],
                    definition=match.get("definition"),
                    similarity=match.get("similarity")
                ))
                if len(alternatives) >= k:
                    break
        if len(alternatives) >= k:
            break
    
    return alternatives[:k]

def _determine_method(llm_result: Dict, search_results: Dict) -> str:
    """Determine which search method found the match."""
    ontology_id = llm_result.get("ontology_id")
    if not ontology_id:
        return "none"
    
    for result_type in ["exact_matches", "fuzzy_matches", "embedding_matches", "fulltext_matches"]:
        for match in search_results.get(result_type, []):
            if match.get("ontology_id") == ontology_id:
                return result_type.replace("_matches", "")
    
    return "unknown" 