#!/usr/bin/env python3
"""
Gradio Web UI for the Universal Ontology Harmonizer Agent
"""

import gradio as gr
import json
import os
import atexit
from dotenv import load_dotenv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.harmonizer_agent import harmonize_metadata, HarmonizedTerm, expand_query_with_llm
from core.ontology_mapper import UnifiedOntologyMapper


# Load environment variables from .env file at the very beginning
load_dotenv()

# Global variables for dynamic ontology loading and persistent DB connection
AVAILABLE_ONTOLOGIES = {}
ONTOLOGY_GROUPS = {}
GLOBAL_MAPPER = None

def get_db_config():
    """Get database configuration from environment variables."""
    return {
        'db_host': os.getenv('DB_HOST', 'localhost'),
        'db_port': int(os.getenv('DB_PORT', '5432')),
        'db_name': os.getenv('DB_NAME', 'ontology_mapper'),
        'db_user': os.getenv('DB_USER', 'postgres'),
        'db_password': os.getenv('DB_PASSWORD', 'Pankyaa@0598820'),  # None if not set
        'embedding_service_url': os.getenv('EMBEDDING_SERVICE_URL', 'http://10.156.4.187:9001')
    }

def initialize_global_mapper():
    """Initialize the global persistent database connection."""
    global GLOBAL_MAPPER
    
    if GLOBAL_MAPPER is not None:
        return GLOBAL_MAPPER
    
    try:
        db_config = get_db_config()
        print(f"🔗 Initializing persistent database connection...")
        print(f"   • Host: {db_config['db_host']}:{db_config['db_port']}")
        print(f"   • Database: {db_config['db_name']}")
        print(f"   • User: {db_config['db_user']}")
        print(f"   • Embedding Service: {db_config['embedding_service_url']}")
        
        GLOBAL_MAPPER = UnifiedOntologyMapper(
            db_host=db_config['db_host'],
            db_port=db_config['db_port'],
            db_name=db_config['db_name'],
            db_user=db_config['db_user'],
            db_password=db_config['db_password'],
            embedding_service_url=db_config['embedding_service_url']
        )
        
        # Test the connection
        stats = GLOBAL_MAPPER.get_stats()
        print(f"✅ Database connection established successfully!")
        print(f"   • Total terms: {stats['total_terms']:,}")
        print(f"   • Terms with embeddings: {stats['terms_with_embeddings']:,}")
        
        # Register cleanup function
        atexit.register(cleanup_global_mapper)
        
        return GLOBAL_MAPPER
        
    except Exception as e:
        print(f"❌ Failed to initialize database connection: {e}")
        print("Please check your database configuration and ensure the database is running.")
        raise

def cleanup_global_mapper():
    """Cleanup function to close the global database connection."""
    global GLOBAL_MAPPER
    if GLOBAL_MAPPER is not None:
        print("🔒 Closing persistent database connection...")
        GLOBAL_MAPPER.close()
        GLOBAL_MAPPER = None

def get_mapper():
    """Get the global mapper instance, initializing if needed."""
    global GLOBAL_MAPPER
    if GLOBAL_MAPPER is None:
        GLOBAL_MAPPER = initialize_global_mapper()
    return GLOBAL_MAPPER

def load_available_ontologies():
    """
    Dynamically load available ontologies from the database.
    Returns a dictionary of ontology codes to their full names.
    """
    global AVAILABLE_ONTOLOGIES, ONTOLOGY_GROUPS
    
    try:
        # Use the global persistent mapper
        mapper = get_mapper()
        
        # Get list of ontologies currently in the database
        available_ontologies = mapper.list_ontologies()
        
        # Get ontology configurations for full names
        ontology_configs = mapper.ontology_configs
        
        # Build the available ontologies dictionary
        AVAILABLE_ONTOLOGIES = {}
        for ont_code in available_ontologies:
            if ont_code in ontology_configs:
                AVAILABLE_ONTOLOGIES[ont_code] = ontology_configs[ont_code]['name']
            else:
                AVAILABLE_ONTOLOGIES[ont_code] = ont_code  # Fallback to code if no name
        
        # Organize ontologies into logical groups based on their purpose
        clinical_disease = ['HPO', 'MONDO', 'DOID', 'ORDO', 'MAXO', 'OGMS']
        cellular_anatomical = ['CL', 'UBERON', 'FMA', 'RADLEX']
        chemical_molecular = ['CHEBI', 'GO', 'PR', 'SO']
        experimental_other = ['EFO', 'OBI', 'PATO', 'NCBITAXON', 'HSAPDV', 'MMUSDV', 
                             'GENO', 'VO', 'GSSO', 'HANCESTRO', 'MPATH']
        
        ONTOLOGY_GROUPS = {
            'clinical': [(f"{ont} - {AVAILABLE_ONTOLOGIES[ont]}", ont) 
                       for ont in clinical_disease if ont in AVAILABLE_ONTOLOGIES],
            'cellular': [(f"{ont} - {AVAILABLE_ONTOLOGIES[ont]}", ont) 
                       for ont in cellular_anatomical if ont in AVAILABLE_ONTOLOGIES],
            'chemical': [(f"{ont} - {AVAILABLE_ONTOLOGIES[ont]}", ont) 
                       for ont in chemical_molecular if ont in AVAILABLE_ONTOLOGIES],
            'other': [(f"{ont} - {AVAILABLE_ONTOLOGIES[ont]}", ont) 
                    for ont in experimental_other if ont in AVAILABLE_ONTOLOGIES]
        }
        
        # Add any ontologies not in predefined groups to 'other'
        all_grouped = set(clinical_disease + cellular_anatomical + chemical_molecular + experimental_other)
        ungrouped = [ont for ont in AVAILABLE_ONTOLOGIES.keys() if ont not in all_grouped]
        for ont in ungrouped:
            ONTOLOGY_GROUPS['other'].append((f"{ont} - {AVAILABLE_ONTOLOGIES[ont]}", ont))
        
        print(f"✅ Loaded {len(AVAILABLE_ONTOLOGIES)} available ontologies from database:")
        for ont_code, ont_name in AVAILABLE_ONTOLOGIES.items():
            print(f"   • {ont_code}: {ont_name}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading ontologies from database: {e}")
        print("Using fallback minimal ontology set...")
        
        # Fallback to minimal set if database connection fails
        AVAILABLE_ONTOLOGIES = {
            'CL': 'Cell Ontology',
            'UBERON': 'Anatomy Ontology',
            'MONDO': 'Disease Ontology',
            'PATO': 'Phenotype and Trait Ontology'
        }
        ONTOLOGY_GROUPS = {
            'clinical': [('MONDO - Disease Ontology', 'MONDO')],
            'cellular': [('CL - Cell Ontology', 'CL'), ('UBERON - Anatomy Ontology', 'UBERON')],
            'chemical': [],
            'other': [('PATO - Phenotype and Trait Ontology', 'PATO')]
        }
        return False

def get_default_selections():
    """Get smart default selections based on available ontologies."""
    defaults = {
        'clinical': [],
        'cellular': [],
        'chemical': [],
        'other': []
    }
    
    # Smart defaults - select common ontologies if available
    preferred_defaults = {
        'clinical': ['MONDO', 'HPO'],
        'cellular': ['CL', 'UBERON'], 
        'chemical': ['CHEBI'],
        'other': ['EFO', 'PATO']
    }
    
    for group, preferred in preferred_defaults.items():
        available_in_group = [ont for _, ont in ONTOLOGY_GROUPS.get(group, [])]
        defaults[group] = [ont for ont in preferred if ont in available_in_group]
    
    return defaults

def query_database_directly(
    dataset_description: str,
    term_name: str,
    selected_ontologies: list,
    search_methods: list,
    top_k: int,
    use_query_expansion: bool,
    num_expanded_queries: int
) -> tuple:
    """
    Query the database directly, with optional query expansion and aggregation.
    """
    if not term_name.strip():
        return {}, "❌ Please enter a term to search for."
    
    if not search_methods:
        return {}, "❌ Please select at least one search method."
    
    try:
        ontology_filter = ",".join(selected_ontologies) if selected_ontologies else None
        mapper = get_mapper()

        if use_query_expansion:
            if not dataset_description.strip():
                print("   [APP-DB] ⚠️ Query expansion skipped: Dataset Description is required.")
                expanded_queries = [term_name]
            else:
                expanded_queries = expand_query_with_llm(term_name, "user query", dataset_description, n=num_expanded_queries)
        else:
            expanded_queries = [term_name]

        print(f"   [APP-DB] Searching with {len(expanded_queries)} variations: {expanded_queries}")
        
        aggregated_results = {method: [] for method in search_methods}
        seen_ids = {method: set() for method in search_methods}
        
        for query in expanded_queries:
            for method in search_methods:
                try:
                    if method == "exact": results = mapper.exact_match(query, ontology_filter)
                    elif method == "fuzzy": results = mapper.fuzzy_match(query, ontology_filter, threshold=0.3)
                    elif method == "fulltext": results = mapper.fulltext_match(query, ontology_filter)
                    elif method == "embedding": results = mapper.embedding_match(query, ontology_filter, threshold=0.5)
                    else: continue
                    
                    for match in results:
                        if match['ontology_id'] not in seen_ids[method]:
                            aggregated_results[method].append(match)
                            seen_ids[method].add(match['ontology_id'])
                except Exception as e:
                    print(f"⚠️ Error in {method} search for query '{query}': {e}")

        # Sort and limit the final aggregated results
        final_results = {}
        total_matches = 0
        for method, matches in aggregated_results.items():
            if method in ["fuzzy", "embedding"] and matches:
                matches.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)
            limited = matches[:top_k]
            final_results[method] = limited
            total_matches += len(limited)
        
        # Create summary text
        summary_lines = [
            f"🔍 **Original Term**: {term_name}",
            f"**Expanded Queries**: {', '.join(expanded_queries)}" if use_query_expansion else "",
            f"📊 **Total Unique Matches Found**: {total_matches} (showing top {top_k} per method)",
            f"🗂️ **Ontologies**: {', '.join(selected_ontologies) if selected_ontologies else 'All'}",
            f"⚙️ **Methods Used**: {', '.join(search_methods)}",
        ]
        
        return final_results, "\n".join(filter(None, summary_lines))
        
    except Exception as e:
        print(f"🔥 UNEXPECTED ERROR in query_database_directly: {e}")
        return {}, f"❌ An unexpected error occurred: {str(e)}"

def run_harmonization(
    dataset_description: str,
    term_name: str,
    field_name: str,
    selected_ontologies: list,
    top_k: int,
    use_query_expansion: bool,
    num_expanded_queries: int
) -> tuple:
    """
    Run the full harmonization workflow: DB search -> AI analysis.
    """
    print("\n" + "="*50)
    print("🚀 [APP] Received request for AI Harmonization")
    print(f"   • Term: '{term_name}'")
    print(f"   • Field: '{field_name}'")
    print(f"   • Ontologies: {selected_ontologies}")
    print(f"   • Description: '{dataset_description[:100]}...'")
    
    if not term_name.strip():
        print("   ⚠️ [APP] Term name is empty. Aborting.")
        return "❌ Please enter a term to harmonize.", {}, "", ""

    # Combine selected ontologies from all groups into a single comma-separated string
    ontology_filter = ",".join(selected_ontologies) if selected_ontologies else None
    
    try:
        print("   🧠 [APP] Calling harmonize_metadata function...")
        # Get the persistent mapper instance
        mapper = get_mapper()
        
        # Call the core harmonization function from the agent
        harmonized_result: HarmonizedTerm = harmonize_metadata(
            dataset_description=dataset_description,
            term_name=term_name,
            field_name=field_name,
            mapper=mapper,
            ontologies=ontology_filter,
            k=top_k,
            use_query_expansion=use_query_expansion,
            num_expanded_queries=num_expanded_queries
        )
        print("   ✅ [APP] Received response from harmonize_metadata.")

        # Check the status from the result object
        if harmonized_result.status == "success":
            print(f"   🎉 [APP] Harmonization successful: {harmonized_result.best_match.ontology_id}")
            # Format results for display - return Python dict, not JSON string
            try:
                best_match_dict = harmonized_result.best_match.model_dump()
                print(f"   [APP] Created result dictionary with {len(best_match_dict)} fields")
            except Exception as e:
                print(f"   [APP] 🔥 Error creating result dict: {e}")
                best_match_dict = {}
            
            # Create comprehensive summary text with all information
            summary_text = (
                f"✅ **Harmonization Successful**\n\n"
                f"**Method**: {harmonized_result.method}\n"
                f"**Confidence**: {harmonized_result.confidence:.2f}\n\n"
                f"**Best Match**: {harmonized_result.best_match.ontology_id} - {harmonized_result.best_match.label}\n"
                f"**Ontology**: {harmonized_result.best_match.ontology}\n\n"
                f"**Definition**: {harmonized_result.best_match.definition or 'No definition available'}\n\n"
                f"**Alternative Matches**: {len(harmonized_result.alternative_matches)} found\n\n"
                f"**Search Summary**:\n"
                f"• Exact: {harmonized_result.search_summary.get('exact_matches', 0)}\n"
                f"• Fuzzy: {harmonized_result.search_summary.get('fuzzy_matches', 0)}\n"
                f"• Semantic: {harmonized_result.search_summary.get('embedding_matches', 0)}\n"
                f"• Full-text: {harmonized_result.search_summary.get('fulltext_matches', 0)}\n\n"
                f"**Reasoning**: {harmonized_result.reasoning}"
            )
            # Return in correct order: ai_json, ai_summary (based on line 665)
            return best_match_dict, summary_text
        else:
            print(f"   ❌ [APP] Harmonization failed or no match found. Status: {harmonized_result.status}")
            summary_text = (
                f"❌ **{harmonized_result.status.replace('_', ' ').title()}**\n\n"
                f"**Search Summary**:\n"
                f"• Exact: {harmonized_result.search_summary.get('exact_matches', 0) if harmonized_result.search_summary else 0}\n"
                f"• Fuzzy: {harmonized_result.search_summary.get('fuzzy_matches', 0) if harmonized_result.search_summary else 0}\n"
                f"• Semantic: {harmonized_result.search_summary.get('embedding_matches', 0) if harmonized_result.search_summary else 0}\n"
                f"• Full-text: {harmonized_result.search_summary.get('fulltext_matches', 0) if harmonized_result.search_summary else 0}\n\n"
                f"**Reasoning**: {harmonized_result.reasoning}"
            )
            return {}, summary_text

    except Exception as e:
        print(f"   🔥 [APP] An unexpected error occurred in the harmonization process: {e}")
        error_summary = f"❌ **Error**: {str(e)}"
        return {}, error_summary

def create_interface():
    custom_css = ".json-container { min-height: 500px; max-height: 500px; overflow-y: auto; } footer { display: none !important; }"
    
    # Ensure ontologies are loaded before creating the interface
    try:
        load_available_ontologies()
    except Exception as e:
        print(f"⚠️  Warning: Could not load ontologies: {e}")
    
    defaults = get_default_selections()
    
    # Get database stats for the status dropdown
    try:
        mapper = get_mapper()
        db_stats = mapper.get_stats()
        total_terms = db_stats['total_terms']
        status_connected = True
        stats_text = f"""**Database Statistics:**
• **Total Terms**: {total_terms:,}
• **Terms with Embeddings**: {db_stats['terms_with_embeddings']:,}
• **Coverage**: {(db_stats['terms_with_embeddings'] / total_terms * 100) if total_terms > 0 else 0:.2f}%
• **Database Size**: {db_stats['database_size']}
**Terms by Ontology:**\n"""
        for ont, count in sorted(db_stats['ontology_counts'].items()):
            stats_text += f"• **{ont}**: {count:,} terms\n"
        
    except Exception as e:
        status_connected = False
        stats_text = f"❌ **Database Connection Failed**\n\nError: {str(e)}"

    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("# 🧬 Universal Ontology Harmonizer")
            with gr.Column(scale=1):
                status_color = "🟢" if status_connected else "🔴"
                status_text_short = f"{status_color} DB Connected" if status_connected else f"{status_color} DB Disconnected"
                with gr.Accordion(status_text_short, open=False):
                    gr.Markdown(stats_text)

        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                # --- Define Input Components FIRST ---
                gr.Markdown("### ⚙️ Search Parameters")
                use_query_expansion_checkbox = gr.Checkbox(label="Enable AI Query Expansion", value=False, info="Requires a Dataset Description", interactive=False)
                num_expansion_slider = gr.Slider(minimum=2, maximum=10, value=3, step=1, label="Number of Expanded Queries", info="How many variations the AI should generate.", visible=False)
                search_methods = gr.CheckboxGroup(choices=["exact", "fuzzy", "embedding", "fulltext"], value=["exact", "embedding"], label="Search Methods (for direct query)")
                top_k = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Top K Results per Method")

                gr.Markdown("### 📝 Harmonization Input")
                dataset_description = gr.Textbox(lines=2, label="Dataset Description", placeholder="e.g., Study of cellular dynamics in neonatal BPD...")
                term_name = gr.Textbox(lines=1, label="Term to Harmonize", placeholder="e.g., AT1, heart disease...")
                field_name = gr.Textbox(lines=1, label="Field Name", placeholder="e.g., cell type, disease...")
                
                with gr.Row():
                    query_db_button = gr.Button("🔍 Query Database Only")
                    run_ai_button = gr.Button("🤖 Run AI Harmonization", variant="primary")
                
                with gr.Row():
                    select_all_ont_btn = gr.Button("Select All Ontologies")
                    clear_all_ont_btn = gr.Button("Clear All Ontologies")
                
                ontology_components = {}
                with gr.Accordion("🩺 Select Ontologies", open=True):
                    with gr.Row():
                        with gr.Column():
                            if ONTOLOGY_GROUPS.get('clinical'): ontology_components['clinical'] = gr.CheckboxGroup(choices=ONTOLOGY_GROUPS['clinical'], value=defaults['clinical'], label="Clinical & Disease")
                            if ONTOLOGY_GROUPS.get('chemical'): ontology_components['chemical'] = gr.CheckboxGroup(choices=ONTOLOGY_GROUPS['chemical'], value=defaults['chemical'], label="Chemical & Molecular")
                        with gr.Column():
                            if ONTOLOGY_GROUPS.get('cellular'): ontology_components['cellular'] = gr.CheckboxGroup(choices=ONTOLOGY_GROUPS['cellular'], value=defaults['cellular'], label="Cellular & Anatomical")
                            if ONTOLOGY_GROUPS.get('other'): ontology_components['other'] = gr.CheckboxGroup(choices=ONTOLOGY_GROUPS['other'], value=defaults['other'], label="Other Ontologies")

                # --- Examples ---
                example_inputs = [
                    dataset_description, term_name, field_name,
                    ontology_components.get('clinical'),
                    ontology_components.get('cellular'),
                    ontology_components.get('chemical'),
                    ontology_components.get('other'),
                    top_k,
                    use_query_expansion_checkbox,
                    num_expansion_slider
                ]
                
                examples = []
                if 'CL' in AVAILABLE_ONTOLOGIES:
                    examples.append([
                        "Study of cellular dynamics in neonatal BPD using samples from preterm infants.",
                        "AT1", "cell type",
                        [], ['CL'], [], [],
                        5, True, 3
                    ])
                if 'MONDO' in AVAILABLE_ONTOLOGIES:
                    examples.append([
                        "Analysis of patient samples with idiopathic pulmonary fibrosis (IPF).",
                        "bacterial pneumonia", "disease",
                        ['MONDO'], [], [], [],
                        5, False, 3
                    ])

                if examples:
                    gr.Examples(examples=examples, inputs=example_inputs, label="💡 Quick Examples")

            with gr.Column(scale=3):
                # --- Define Output Components ---
                gr.Markdown("### 📊 Results")
                with gr.Tabs():
                    with gr.TabItem("🔍 Database Query Results"):
                        db_summary = gr.Markdown("")
                        db_json = gr.JSON(label="Raw Database Results", show_label=True, elem_classes=["json-container"])
                    with gr.TabItem("🤖 AI Harmonization Results"):
                        ai_summary = gr.Markdown("")
                        ai_json = gr.JSON(label="Harmonized AI Result", show_label=True, elem_classes=["json-container"])

        # --- Define Event Logic ---
        def combine_selected_ontologies(clinical, cellular, chemical, other):
            return (clinical or []) + (cellular or []) + (chemical or []) + (other or [])

        def toggle_query_expansion_controls(description_text, current_value):
            is_enabled = bool(description_text and description_text.strip())
            # If the description is cleared, we disable and uncheck the box.
            if not is_enabled:
                return gr.update(interactive=False, value=False)
            # If description is present, we enable the box but don't change its checked status.
            return gr.update(interactive=True, value=current_value)

        def run_database_query(desc, term, clin, cell, chem, other, methods, k, use_exp, num_exp):
            selected = combine_selected_ontologies(clin, cell, chem, other)
            return query_database_directly(desc, term, selected, methods, k, use_exp, num_exp)

        def run_ai_harmonization(desc, term, field, clin, cell, chem, other, k, use_exp, num_exp):
            selected = combine_selected_ontologies(clin, cell, chem, other)
            return run_harmonization(desc, term, field, selected, k, use_exp, num_exp)

        def toggle_expansion_slider(is_enabled): return gr.update(visible=is_enabled)
        
        all_ont_comps = [ontology_components.get(g) for g in ['clinical', 'chemical', 'cellular', 'other'] if ONTOLOGY_GROUPS.get(g)]
        def select_all(): return tuple([ont for _, ont in ONTOLOGY_GROUPS.get(g, [])] for g in ['clinical', 'chemical', 'cellular', 'other'])
        def clear_all(): return tuple([] for _ in range(4))

        # --- Wire Up Events LAST ---
        dataset_description.change(
            fn=toggle_query_expansion_controls,
            inputs=[dataset_description, use_query_expansion_checkbox],
            outputs=use_query_expansion_checkbox
        )
        use_query_expansion_checkbox.change(fn=toggle_expansion_slider, inputs=use_query_expansion_checkbox, outputs=num_expansion_slider)

        # Only wire up events if components exist
        if query_db_button and db_json and db_summary:
            query_db_button.click(
                fn=run_database_query,
                inputs=[dataset_description, term_name, ontology_components.get('clinical'), ontology_components.get('cellular'), ontology_components.get('chemical'), ontology_components.get('other'), search_methods, top_k, use_query_expansion_checkbox, num_expansion_slider],
                outputs=[db_json, db_summary]
            )
        
        if run_ai_button and ai_json and ai_summary:
            run_ai_button.click(
                fn=run_ai_harmonization,
                inputs=[dataset_description, term_name, field_name, ontology_components.get('clinical'), ontology_components.get('cellular'), ontology_components.get('chemical'), ontology_components.get('other'), top_k, use_query_expansion_checkbox, num_expansion_slider],
                outputs=[ai_json, ai_summary]
            )
        if all_ont_comps:
            select_all_ont_btn.click(fn=select_all, outputs=all_ont_comps)
            clear_all_ont_btn.click(fn=clear_all, outputs=all_ont_comps)
    return demo

# Create the app instance for external imports (only when needed)
app = None

def get_app():
    """Get the app instance, creating it if needed."""
    global app
    if app is None:
        app = create_interface()
    return app

if __name__ == "__main__":
    print("🚀 Starting Universal Ontology Harmonizer Web App...")
    
    # Initialize the persistent database connection at startup
    try:
        initialize_global_mapper()
        print("📋 Loading available ontologies...")
        load_available_ontologies()
        print("✅ App initialization complete!")
    except Exception as e:
        print(f"❌ Failed to initialize app: {e}")
        print("The app will still start but may have limited functionality.")
    
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False) 