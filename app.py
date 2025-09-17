#!/usr/bin/env python3
"""Streamlit web interface for LLM PMID Checker."""
import streamlit as st
import asyncio
import logging
import pandas as pd
import plotly.express as px
import requests
from typing import List
from datetime import datetime
import json

# Import project modules
from src.triple_evaluator import TripleEvaluatorSystem, TripleEvaluationResult
from src.node_normalization import NodeNormalizationClient
from src.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="LLM PMID Checker",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_environment():
    """Check if the environment is properly configured."""
    missing_vars = []
    warnings = []
    
    # Check required environment variables
    if not settings.ncbi_email:
        missing_vars.append("NCBI_EMAIL")
    if not settings.ncbi_api_key:
        warnings.append("NCBI_API_KEY (optional but recommended for higher rate limits)")
    
    # Check Ollama connection
    try:
        response = requests.get(f"{settings.ollama_base_url}/api/version", timeout=5)
        ollama_status = "✅ Connected" if response.status_code == 200 else "❌ Error"
    except Exception as e:
        ollama_status = f"❌ Not accessible ({str(e)[:50]}...)"
    
    return {
        "missing_vars": missing_vars,
        "warnings": warnings,
        "ollama_status": ollama_status
    }

def display_environment_status():
    """Display environment configuration status."""
    status = check_environment()
    
    st.sidebar.header("🔧 Environment Status")
    
    # Ollama status
    st.sidebar.write(f"**Ollama Server**: {status['ollama_status']}")
    st.sidebar.write(f"**Base URL**: {settings.ollama_base_url}")
    
    # Missing variables
    if status["missing_vars"]:
        st.sidebar.error(f"❌ Missing: {', '.join(status['missing_vars'])}")
    else:
        st.sidebar.success("✅ Required variables set")
    
    # Warnings
    if status["warnings"]:
        for warning in status["warnings"]:
            st.sidebar.warning(f"⚠️ {warning}")
    

async def run_evaluation(triple_data: dict, pmids: List[str], model: str, progress_callback=None) -> TripleEvaluationResult:
    """Run the triple check asynchronously with progress tracking."""
    try:
        if progress_callback:
            progress_callback(10, "🔧 Initializing checking system...")
        
        evaluator = TripleEvaluatorSystem(llm_provider=model)
        
        if progress_callback:
            progress_callback(20, "📡 Fetching abstracts from PubMed...")
        
        # Get abstracts first
        abstract_data = evaluator.pmid_extractor.extract_abstracts(pmids)
        
        # Entity normalization is already done in the UI, so skip that step
        if progress_callback:
            progress_callback(40, "🤖 Starting AI model evaluation...")
        
        # Manual evaluation with progress tracking for each PMID
        from src.evaluation_agent import TripleData, TripleEvaluation
        from src.triple_evaluator import TripleEvaluationResult
        
        # Create enriched triple data
        triple_obj = TripleData(
            subject=triple_data['subject'],
            predicate=triple_data['predicate'],
            object=triple_data['object'],
            subject_names=triple_data.get('subject_names'),
            object_names=triple_data.get('object_names')
        )
        
        evaluations = []
        
        # Process each PMID with individual progress updates
        for i, pmid in enumerate(pmids):
            data = abstract_data.get(pmid)
            if not data:
                # PMID not found in results
                evaluations.append(TripleEvaluation(
                    pmid=pmid,
                    is_supported=False,
                    supporting_sentence=None,
                    confidence=0.0,
                    reasoning="PMID not found in results"
                ))
                continue
                
            if data.error:
                # Error extracting abstract
                evaluations.append(TripleEvaluation(
                    pmid=pmid,
                    is_supported=False,
                    supporting_sentence=None,
                    confidence=0.0,
                    reasoning=f"Error: {data.error}"
                ))
                continue
            
            if not data.abstract.strip():
                # No abstract available
                evaluations.append(TripleEvaluation(
                    pmid=pmid,
                    is_supported=False,
                    supporting_sentence=None,
                    confidence=0.0,
                    reasoning="No abstract available"
                ))
                continue
            
            # Evaluate valid PMID with progress update
            if progress_callback:
                current_progress = 40 + int((i / len(pmids)) * 50)
                progress_callback(current_progress, f"🔬 Evaluating PMID {pmid} ({i+1}/{len(pmids)})...")
            
            try:
                evaluation = await evaluator.evaluation_agent.evaluate_triple_against_abstract(
                    triple=triple_obj,
                    abstract=data.abstract,
                    pmid=pmid,
                    title=data.title
                )
                
                # Apply validation rules
                evaluation = evaluator._validate_evaluation_logic(evaluation, pmid)
                evaluations.append(evaluation)
                
            except Exception as e:
                evaluations.append(TripleEvaluation(
                    pmid=pmid,
                    is_supported=False,
                    supporting_sentence=None,
                    confidence=0.0,
                    reasoning=f"Evaluation failed: {str(e)}"
                ))
        
        if progress_callback:
            progress_callback(90, "📊 Finalizing results...")
        
        # Sort evaluations by original PMID order
        pmid_order = {pmid: idx for idx, pmid in enumerate(pmids)}
        evaluations.sort(key=lambda x: pmid_order.get(x.pmid, float('inf')))
        
        result = TripleEvaluationResult(triple=triple_obj, evaluations=evaluations)
        
        
        if progress_callback:
            progress_callback(100, "✅ Evaluation completed!")
        
        return result
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise e

def create_results_dataframe(result: TripleEvaluationResult) -> pd.DataFrame:
    """Create a pandas DataFrame from evaluation results."""
    data = []
    for eval_result in result.evaluations:
        data.append({
            'PMID': f"https://pubmed.ncbi.nlm.nih.gov/{eval_result.pmid}",
            'Supported': eval_result.is_supported,
            'Confidence': eval_result.confidence,
            'Supporting Sentence': eval_result.supporting_sentence or ''
        })
    return pd.DataFrame(data)

def create_summary_charts(df: pd.DataFrame):
    """Create summary visualizations."""
    # Pie chart of support distribution
    support_counts = df['Supported'].value_counts()
    
    fig_pie = px.pie(
        values=support_counts.values,
        names=['Supported' if x else 'Not Supported' for x in support_counts.index],
        title="Support Distribution",
        color_discrete_map={
            'Supported': '#2E8B57',
            'Not Supported': '#DC143C'
        }
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie)

def main():
    """Main Streamlit application."""
    st.title("🧬 LLM PMID Checker System")
    st.markdown("""
    Check whether research triples are supported by PubMed abstracts using large language models with **enhanced semantic understanding**.
    
    **How it works:**
    1. **Choose input format**: Entity names (with automatic normalization) or direct CURIEs
    2. **Enter a research triple** (e.g., *SIX1 affects Cell Proliferation* or *NCBIGene:6495 affects UMLS:C0596290*)
    3. **Entity normalization**: System automatically finds equivalent names and synonyms
    4. **Provide PubMed IDs** to evaluate against
    5. **Choose an AI model** for evaluation with enriched semantic context
    6. **Get enhanced results** with confidence scores, supporting evidence, and semantic insights
    
    **🆕 New Features:**
    - **Dual Input Modes**: Support for both entity names and semantic identifiers (CURIEs)
    - **Automatic Normalization**: Finds equivalent names using ARAX and SRI Node Normalization APIs
    - **Enhanced AI Evaluation**: LLM considers all equivalent names for better accuracy
    """)
    
    # Display environment status
    display_environment_status()
    
    # Main interface
    with st.container():
        st.header("📝 Input Parameters")
        
        # Triple input
        col1, col2, col3 = st.columns(3)
        with col1:
            subject = st.text_input("Subject", value="SIX1", help="The subject of the research triple")
        with col2:
            predicate = st.text_input("Predicate", value="affects", help="The relationship/action")
        with col3:
            object_ = st.text_input("Object", value="Cell Proliferation", help="The object of the research triple")
        
        # PMID input options
        st.subheader("📄 PMID Input")
        
        pmid_input_method = st.radio(
            "Choose input method:",
            ["Manual Entry", "File Upload"],
            horizontal=True
        )
        
        pmids = []
        
        if pmid_input_method == "Manual Entry":
            pmid_text = st.text_area(
                "Enter PMIDs (space, comma, or newline separated)",
                value="34513929, 16488997, 14695375, 23613228, 34561318, 28473774, 26175950",
                height=100,
                help="Enter PubMed IDs separated by spaces, commas, or newlines"
            )
            
            if pmid_text.strip():
                # Parse PMIDs from text - handle spaces, commas, and newlines
                pmid_text = pmid_text.replace(',', ' ').replace('\n', ' ')
                pmids = [pmid.strip() for pmid in pmid_text.split() if pmid.strip()]
        
        else:  # File Upload
            uploaded_file = st.file_uploader(
                "Upload PMIDs file",
                type=['txt'],
                help="Upload a text file with one PMID per line"
            )
            
            if uploaded_file is not None:
                pmids = [line.decode('utf-8').strip() for line in uploaded_file.readlines() if line.decode('utf-8').strip()]
        
        # Display parsed PMIDs
        if pmids:
            st.info(f"📊 Found {len(pmids)} PMIDs: {', '.join(pmids[:10])}{' ...' if len(pmids) > 10 else ''}")
        
        # Model selection
        st.subheader("🤖 Model Selection")
        model = st.selectbox(
            "Choose AI Model",
            options=["hermes4", "gpt-oss"],
            index=0,
            help="Select the language model for evaluation"
        )
        
        # Model info
        model_info = {
            "hermes4": "Hermes 4 70B (Q4_K_XL) - Latest model with hybrid reasoning, ~42GB VRAM",
            "gpt-oss": "GPT-OSS 120B - Requires ~65GB VRAM, higher accuracy but slower"
        }
        st.info(f"ℹ️ {model_info[model]}")
        
        # Evaluation button
        st.divider()
        
        if st.button("🔬 Start Evaluation", type="primary", disabled=not (subject and predicate and object_ and pmids)):
            if not subject or not predicate or not object_:
                st.error("Please provide all three parts of the triple (Subject, Predicate, Object)")
                return
            
            if not pmids:
                st.error("Please provide at least one PMID")
                return
            
            # Check environment
            env_status = check_environment()
            if env_status["missing_vars"]:
                st.error(f"Missing required environment variables: {', '.join(env_status['missing_vars'])}")
                st.info("Please configure your .env file as shown in the sidebar")
                return
            
            # Create enriched triple data with normalization
            normalization_client = NodeNormalizationClient()
            subject_names = normalization_client.get_equivalent_names(name=subject) or [subject]
            object_names = normalization_client.get_equivalent_names(name=object_) or [object_]
            
            triple_data = {
                'subject': subject,
                'predicate': predicate,
                'object': object_,
                'subject_names': subject_names,
                'object_names': object_names
            }
            
            with st.spinner(f"🔍 Checking triple **{subject} {predicate} {object_}** against {len(pmids)} PMIDs..."):
                try:
                    # Run async evaluation with enriched data
                    result = asyncio.run(run_evaluation(triple_data, pmids, model))
                    
                    # Store results in session state
                    st.session_state.last_result = result
                    st.session_state.last_triple = [subject, predicate, object_]
                    st.session_state.last_model = model
                    
                except Exception as e:
                    st.error(f"❌ Evaluation failed: {str(e)}")
                    st.info("Make sure Ollama is running and the model is available")
                    return
    
    # Display results if available
    if hasattr(st.session_state, 'last_result') and st.session_state.last_result:
        st.divider()
        display_results(st.session_state.last_result, st.session_state.last_triple, st.session_state.last_model)

def display_results(result: TripleEvaluationResult, triple: List[str], model: str):
    """Display evaluation results with visualizations."""
    st.header("📊 Evaluation Results")
    
    # Summary metrics
    summary = result.get_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total PMIDs", summary['total_pmids'])
    with col2:
        st.metric("Supported", summary['supported_pmids'], f"{summary['supported_percentage']}%")
    with col3:
        st.metric("Not Supported", summary['unsupported_pmids'], f"{summary['unsupported_percentage']}%")
    with col4:
        avg_confidence = pd.DataFrame([
            {'confidence': eval_result.confidence} 
            for eval_result in result.evaluations 
            if eval_result.is_supported
        ])['confidence'].mean() if result.evaluations else 0
        st.metric("Avg Confidence", f"{avg_confidence:.2f}" if not pd.isna(avg_confidence) else "N/A")
    
    # Create DataFrame for detailed results
    df = create_results_dataframe(result)
    
    # Charts
    if not df.empty:
        create_summary_charts(df)
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        show_supported = st.checkbox("Show Supported", value=True)
    with col2:
        show_not_supported = st.checkbox("Show Not Supported", value=True)
    
            # Filter dataframe
        filtered_df = df.copy()
        if not show_supported:
            filtered_df = filtered_df[~filtered_df['Supported']]
        if not show_not_supported:
            filtered_df = filtered_df[filtered_df['Supported']]
    
    # Display filtered results
    if not filtered_df.empty:
        # Style the dataframe
        def style_row(row):
            if row['Supported']:
                return ['background-color: #e8f5e8'] * len(row)
            else:
                return ['background-color: #fde8e8'] * len(row)
        
        styled_df = filtered_df.style.apply(style_row, axis=1)
        st.dataframe(styled_df, height=400)
        
        # Export options
        st.subheader("📥 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download as CSV
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name=f"triple_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download as JSON
            json_data = {
                "triple": {
                    "subject": triple[0],
                    "predicate": triple[1],
                    "object": triple[2]
                },
                "model": model,
                "timestamp": datetime.now().isoformat(),
                "summary": summary,
                "evaluations": [
                    {
                        "pmid": eval_result.pmid,
                        "is_supported": eval_result.is_supported,
                        "confidence": eval_result.confidence,
                        "supporting_sentence": eval_result.supporting_sentence
                    }
                    for eval_result in result.evaluations
                ]
            }
            
            st.download_button(
                label="📋 Download JSON",
                data=json.dumps(json_data, indent=2),
                file_name=f"triple_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col3:
            # Copy CLI format
            cli_output = result.format_output()
            st.download_button(
                label="🖥️ Download CLI Format",
                data=cli_output,
                file_name=f"triple_evaluation_cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    else:
        st.warning("No results match the current filters.")
    
    # Detailed view for each evaluation
    st.subheader("🔍 Detailed Evaluation")
    
    # Select PMID for detailed view
    pmids_for_detail = [eval_result.pmid for eval_result in result.evaluations]
    selected_pmid = st.selectbox("Select PMID for detailed view:", pmids_for_detail)
    
    if selected_pmid:
        selected_eval = next((eval_result for eval_result in result.evaluations if eval_result.pmid == selected_pmid), None)
        if selected_eval:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("**PMID:**", selected_eval.pmid)
                st.write("**Supported:**", "✅ Yes" if selected_eval.is_supported else "❌ No")
                st.write("**Confidence:**", f"{selected_eval.confidence:.2f}")
                
                # Link to PubMed
                st.markdown(f"[📖 View on PubMed](https://pubmed.ncbi.nlm.nih.gov/{selected_eval.pmid})")
            
            with col2:
                st.write("**Most Likely Supporting Sentence:**")
                if selected_eval.supporting_sentence:
                    st.success(selected_eval.supporting_sentence)
                else:
                    st.write("_No supporting sentence provided_")

def run_app():
    """Main Streamlit app."""
    
    # Header
    st.markdown("""
    <div style="padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; text-align: center;">🧬 LLM PMID Checker</h1>
        <p style="color: white; margin: 0.5rem 0 0 0; text-align: center;">
            Check research triples against PubMed abstracts using AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check environment and display status
    display_environment_status()
    
    # Main content
    main_container()

def main_container():
    """Main application container."""
    
    # Triple input section
    st.header("🔬 Research Triple")
    st.markdown("Enter the research relationship you want to check:")
    
    # Input mode selection
    input_mode = st.radio(
        "Choose input format:",
        ["Entity Names (with normalization)", "CURIEs (direct identifiers)"],
        help="Entity Names: Automatically finds equivalent names (e.g., 'SIX1'). CURIEs: Direct semantic identifiers (e.g., 'NCBIGene:6495')",
        horizontal=True
    )
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        if input_mode == "Entity Names (with normalization)":
            subject = st.text_input("Subject", value="SIX1", help="Gene/protein name (will be normalized)")
        else:
            subject = st.text_input("Subject CURIE", value="NCBIGene:6495", help="Subject CURIE (e.g., NCBIGene:6495)")
    with col2:
        predicate = st.text_input("Predicate", value="affects", help="The relationship or action")
    with col3:
        if input_mode == "Entity Names (with normalization)":
            object_ = st.text_input("Object", value="Cell Proliferation", help="Process/condition name (will be normalized)")
        else:
            object_ = st.text_input("Object CURIE", value="UMLS:C0596290", help="Object CURIE (e.g., UMLS:C0596290)")
    
    # Initialize triple data structure
    triple_data = None
    
    if subject and predicate and object_:
        st.success(f"Triple: **{subject}** {predicate} **{object_}**")
        
        # Perform entity normalization if needed
        if input_mode == "Entity Names (with normalization)":
            with st.spinner("🧬 Normalizing entities..."):
                normalization_client = NodeNormalizationClient()
                
                # Get equivalent names
                subject_names = normalization_client.get_equivalent_names(name=subject)
                object_names = normalization_client.get_equivalent_names(name=object_)
                
                if not subject_names:
                    st.warning(f"⚠️ No equivalent names found for subject: {subject}")
                    subject_names = [subject]
                if not object_names:
                    st.warning(f"⚠️ No equivalent names found for object: {object_}")
                    object_names = [object_]
                
                # Show equivalent names found (always show the section for transparency)
                with st.expander("🔍 Equivalent Names Found"):
                    st.write(f"**{subject}** equivalent names:")
                    if len(subject_names) > 1:
                        for name in subject_names[:5]:  # Show first 5
                            st.write(f"• {name}")
                        if len(subject_names) > 5:
                            st.write(f"... and {len(subject_names) - 5} more")
                    else:
                        st.write(f"• {subject_names[0]} (primary name)")
                    
                    st.write(f"**{object_}** equivalent names:")
                    if len(object_names) > 1:
                        for name in object_names[:5]:  # Show first 5
                            st.write(f"• {name}")
                        if len(object_names) > 5:
                            st.write(f"... and {len(object_names) - 5} more")
                    else:
                        st.write(f"• {object_names[0]} (primary name)")
                
                triple_data = {
                    'subject': subject,
                    'predicate': predicate,
                    'object': object_,
                    'subject_names': subject_names,
                    'object_names': object_names
                }
        else:
            # CURIE mode - get equivalent names from CURIEs
            with st.spinner("🧬 Normalizing CURIEs..."):
                normalization_client = NodeNormalizationClient()
                
                # Get equivalent names from CURIEs
                subject_names = normalization_client.get_equivalent_names(curie=subject)
                object_names = normalization_client.get_equivalent_names(curie=object_)
                
                if not subject_names:
                    st.warning(f"⚠️ No equivalent names found for subject CURIE: {subject}")
                    subject_names = [subject]
                if not object_names:
                    st.warning(f"⚠️ No equivalent names found for object CURIE: {object_}")
                    object_names = [object_]
                
                # Show primary names and equivalents
                primary_subject = subject_names[0] if subject_names else subject
                primary_object = object_names[0] if object_names else object_
                
                st.info(f"🏷️ Resolved to: **{primary_subject}** {predicate} **{primary_object}**")
                
                # Always show equivalent names section for transparency
                with st.expander("🔍 Equivalent Names Found"):
                    st.write(f"**{subject}** equivalent names:")
                    if len(subject_names) > 1:
                        for name in subject_names[:5]:
                            st.write(f"• {name}")
                        if len(subject_names) > 5:
                            st.write(f"... and {len(subject_names) - 5} more")
                    else:
                        st.write(f"• {subject_names[0]} (primary name)")
                    
                    st.write(f"**{object_}** equivalent names:")
                    if len(object_names) > 1:
                        for name in object_names[:5]:
                            st.write(f"• {name}")
                        if len(object_names) > 5:
                            st.write(f"... and {len(object_names) - 5} more")
                    else:
                        st.write(f"• {object_names[0]} (primary name)")
                
                triple_data = {
                    'subject': primary_subject,
                    'predicate': predicate,
                    'object': primary_object,
                    'subject_names': subject_names,
                    'object_names': object_names
                }
    
    st.divider()
    
    # PMID input section
    st.header("📄 PubMed IDs")
    
    input_tab1, input_tab2 = st.tabs(["📝 Manual Entry", "📁 File Upload"])
    
    pmids = []
    
    with input_tab1:
        pmid_input = st.text_area(
            "Enter PMIDs",
            value="34513929\n16488997\n14695375\n23613228\n34561318\n28473774\n26175950",
            height=150,
            help="Enter PubMed IDs separated by spaces, commas, or newlines"
        )
        
        if pmid_input.strip():
            # Parse PMIDs - handle spaces, commas, and newlines
            pmid_text = pmid_input.replace(',', ' ').replace('\n', ' ')
            pmids = [pmid.strip() for pmid in pmid_text.split() if pmid.strip()]
    
    with input_tab2:
        uploaded_file = st.file_uploader(
            "Upload PMIDs file",
            type=['txt'],
            help="Upload a text file with one PMID per line"
        )
        
        if uploaded_file is not None:
            content = uploaded_file.read().decode('utf-8')
            pmids = [line.strip() for line in content.split('\n') if line.strip()]
            st.success(f"Loaded {len(pmids)} PMIDs from file")
    
    # Display PMID preview
    if pmids:
        with st.expander(f"📊 Preview PMIDs ({len(pmids)} total)"):
            # Show first 20 PMIDs
            display_pmids = pmids[:20]
            cols = st.columns(5)
            for i, pmid in enumerate(display_pmids):
                with cols[i % 5]:
                    st.code(pmid)
            
            if len(pmids) > 20:
                st.info(f"... and {len(pmids) - 20} more PMIDs")
    
    st.divider()
    
    # Model selection
    st.header("🤖 AI Model Selection")
    
    col1, col2 = st.columns(2)
    with col1:
        model = st.selectbox(
            "Select Model",
            options=["hermes4", "gpt-oss"],
            index=0,
            help="Choose the language model for evaluation"
        )
    
    with col2:
        # Model information
        if model == "hermes4":
            st.info("🧠 **Hermes 4 70B (Q4_K_XL)**\n- ~42GB VRAM required\n- Hybrid reasoning mode with <think> tags\n- Enhanced JSON schema adherence\n- Improved instruction following")
        else:
            st.info("🧠 **GPT-OSS 120B**\n- ~65GB VRAM required\n- Higher accuracy\n- Slower processing")
    
    
    st.divider()
    
    # Evaluation section
    can_evaluate = triple_data is not None and pmids
    
    if can_evaluate:
        if st.button("🚀 Start Check", type="primary"):
            
            # Check environment first
            env_status = check_environment()
            if env_status["missing_vars"]:
                st.error(f"❌ Missing required environment variables: {', '.join(env_status['missing_vars'])}")
                with st.expander("📋 Setup Instructions"):
                    st.markdown("""
                    **Create a .env file with:**
                    ```
                    NCBI_EMAIL=your.email@example.com
                    NCBI_API_KEY=your_api_key_here
                    OLLAMA_BASE_URL=http://localhost:11434
                    ```
                    """)
                return
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Progress callback function
            def update_progress(progress_value, message):
                progress_bar.progress(progress_value)
                status_text.text(message)
            
            try:
                # Run the evaluation with progress tracking
                result = asyncio.run(run_evaluation(triple_data, pmids, model, update_progress))
                
                # Store in session state
                st.session_state.last_result = result
                st.session_state.last_triple = [triple_data['subject'], triple_data['predicate'], triple_data['object']]
                st.session_state.last_model = model
                
                # Clear progress indicators after a short delay
                import time
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
                # Rerun to show results
                st.rerun()
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Evaluation failed: {str(e)}")
                
                # Detailed error information
                with st.expander("🐛 Error Details"):
                    st.code(str(e))
                    st.markdown("**Troubleshooting:**")
                    st.markdown("- Make sure Ollama is running: `./setup_ollama_pmid_support.sh`")
                    st.markdown("- Check if the model is available: `ollama list`")
                    st.markdown("- Verify your .env configuration")
    
    else:
        st.info("👆 Please fill in all fields to start check")
    
    # Display results if available
    if hasattr(st.session_state, 'last_result') and st.session_state.last_result:
        st.divider()
        display_evaluation_results()

def display_evaluation_results():
    """Display the evaluation results with charts and detailed breakdown."""
    result = st.session_state.last_result
    triple = st.session_state.last_triple
    model = st.session_state.last_model
    
    st.header("📈 Results")
    
    # Summary
    summary = result.get_summary()
    st.subheader(f"Triple: **{triple[0]}** {triple[1]} **{triple[2]}** (Model: {model})")
    
    # Layout: 2x2 metrics on left, pie chart on right
    metrics_col, chart_col = st.columns([1, 1])
    
    with metrics_col:
        # Top row of metrics
        metric_row1_col1, metric_row1_col2 = st.columns(2)
        with metric_row1_col1:
            st.metric("Total PMIDs", summary['total_pmids'])
        with metric_row1_col2:
            st.metric("✅ Supported", summary['supported_pmids'], delta=f"{summary['supported_percentage']}%")
        
        # Bottom row of metrics
        metric_row2_col1, metric_row2_col2 = st.columns(2)
        with metric_row2_col1:
            st.metric("❌ Not Supported", summary['unsupported_pmids'], delta=f"{summary['unsupported_percentage']}%")
        with metric_row2_col2:
            # Calculate average confidence for supported PMIDs
            supported_evals = [e for e in result.evaluations if e.is_supported]
            avg_conf = sum(e.confidence for e in supported_evals) / len(supported_evals) if supported_evals else 0
            st.metric("Avg Confidence", f"{avg_conf:.2f}")
    
    with chart_col:
        # Create and display pie chart
        df = create_results_dataframe(result)
        if not df.empty:
            create_summary_charts(df)
    
    # Results table with tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 All Results", "✅ Supported Only", "❌ Not Supported Only"])
    
    with tab1:
        display_results_table(df, "all")
    
    with tab2:
        supported_df = df[df['Supported']]
        if not supported_df.empty:
            display_results_table(supported_df, "supported")
        else:
            st.info("No supported PMIDs found.")
    
    with tab3:
        not_supported_df = df[~df['Supported']]
        if not not_supported_df.empty:
            display_results_table(not_supported_df, "not_supported")
        else:
            st.info("All PMIDs are supported!")
    
    # Export section
    st.subheader("📥 Export Results")
    export_results(result, triple, model)

def display_results_table(df: pd.DataFrame, view_type: str):
    """Display results table with appropriate styling."""
    if df.empty:
        st.info("No results to display.")
        return
    
    # Configure columns
    column_config = {
        'PMID': st.column_config.LinkColumn('PMID', help="PubMed ID - Click to view on PubMed", display_text="https://pubmed.ncbi.nlm.nih.gov/(.+)"),
        'Supported': st.column_config.CheckboxColumn('Supported', help="Whether the triple is supported"),
        'Confidence': st.column_config.ProgressColumn('Confidence', help="Confidence score", min_value=0, max_value=1),
        'Most Likely Supporting Sentence': st.column_config.TextColumn('Most Likely Supporting Sentence', help="Key supporting sentence from abstract", width="large")
    }
    
    # Display with styling
    st.dataframe(
        df,
        column_config=column_config,
        hide_index=True,
        height=min(400, len(df) * 35 + 50)
    )

def export_results(result: TripleEvaluationResult, triple: List[str], model: str):
    """Provide export options for results."""
    col1, col2, col3 = st.columns(3)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with col1:
        # CSV export
        df = create_results_dataframe(result)
        csv = df.to_csv(index=False)
        st.download_button(
            label="📄 Download as CSV",
            data=csv,
            file_name=f"pmid_evaluation_{timestamp}.csv",
            mime="text/csv",
        )
    
    with col2:
        # JSON export
        summary = result.get_summary()
        json_data = {
            "metadata": {
                "triple": {"subject": triple[0], "predicate": triple[1], "object": triple[2]},
                "model": model,
                "timestamp": datetime.now().isoformat(),
                "total_pmids": summary['total_pmids']
            },
            "summary": summary,
            "evaluations": [
                {
                    "pmid": e.pmid,
                    "is_supported": e.is_supported,
                    "confidence": e.confidence,
                    "supporting_sentence": e.supporting_sentence
                }
                for e in result.evaluations
            ]
        }
        
        st.download_button(
            label="📋 Download as JSON",
            data=json.dumps(json_data, indent=2),
            file_name=f"pmid_evaluation_{timestamp}.json",
            mime="application/json",
        )
    
    with col3:
        # CLI format export
        cli_output = result.format_output()
        st.download_button(
            label="🖥️ CLI Format",
            data=cli_output,
            file_name=f"pmid_evaluation_{timestamp}.txt",
            mime="text/plain",
        )

if __name__ == "__main__":
    run_app()