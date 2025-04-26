import openai
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import os
import json
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import numpy as np
from faker import Faker
import plotly.express as px
import time
from typing import List, Dict, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import zipfile
import io

# Custom CSS for enhanced UI
def inject_custom_css():
    st.markdown("""
    <style>
        /* Main container styling */
        .stApp {
            background-color: #f5f7fa;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2c3e50 0%, #1a2530 100%);
            color: white;
        }
        
        /* Sidebar headers */
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: white !important;
        }
        
        /* Button styling - consistent sizing */
        .stButton>button {
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.2s;
            min-width: 120px;
            height: 40px;
        }
        
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        /* Primary button */
        div[data-testid="stButton"]:has(button[kind="primary"]) button {
            background: linear-gradient(90deg, #6e48aa 0%, #9d50bb 100%);
            border: none;
            color: white;
        }
        
        /* Secondary button */
        div[data-testid="stButton"]:has(button[kind="secondary"]) button {
            background: linear-gradient(90deg, #4776E6 0%, #8E54E9 100%);
            border: none;
            color: white;
        }
        
        /* Text input styling */
        .stTextArea textarea, .stTextInput input {
            border-radius: 8px !important;
            padding: 10px !important;
            border: 1px solid #e1e4e8 !important;
        }
        
        /* Card styling */
        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        
        /* Metric cards */
        [data-testid="metric-container"] {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
        }
        
        /* Progress bar */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #6e48aa 0%, #9d50bb 100%);
        }
        
        /* Tab styling */
        [role="tab"] {
            padding: 8px 16px !important;
            border-radius: 8px !important;
            margin: 0 4px !important;
        }
        
        [role="tab"][aria-selected="true"] {
            background: linear-gradient(90deg, #6e48aa 0%, #9d50bb 100%);
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize with error handling
def initialize_app():
    try:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("❌ Missing OPENAI_API_KEY in .env file")
            st.stop()

        client = OpenAI(api_key=api_key)
        
        chroma_client = chromadb.PersistentClient(
            path="storage/chroma",
            settings=chromadb.Settings(allow_reset=True)
        )
        
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small"
        )
        
        collection = chroma_client.get_or_create_collection(
            name="documentation",
            embedding_function=openai_ef
        )
        
        return client, collection, Faker()
    
    except Exception as e:
        st.error(f"❌ Initialization failed: {str(e)}")
        st.stop()

# Initialize services
client, collection, fake = initialize_app()

def visualize_document_similarity():
    """Document embeddings visualization with original style"""
    try:
        # Get all embeddings and metadata
        result = collection.get(include=["embeddings", "metadatas"])
        embeddings = result["embeddings"]
        metadatas = result.get("metadatas", [{}]*len(embeddings))
        
        # Check if we have enough data
        if embeddings is None or len(embeddings) < 2:
            st.info("""
            <div class="card">
                <h3>📊 Visualization Unavailable</h3>
                <p>You need at least 2 documents to visualize similarity.</p>
                <p>Try generating sample data or adding your own documents.</p>
            </div>
            """, unsafe_allow_html=True)
            return

        # Convert to numpy array and validate
        embeddings_array = np.array(embeddings)
        if embeddings_array.size == 0:
            st.warning("No embeddings found for visualization")
            return

        # Dimensionality reduction
        reduction_method = st.radio(
            "Visualization Method",
            ["PCA", "t-SNE"],
            horizontal=True
        )
        
        if reduction_method == "PCA":
            reducer = PCA(n_components=2)
        else:
            reducer = TSNE(n_components=2, perplexity=min(30, len(embeddings)-1))
            
        reduced_embeddings = reducer.fit_transform(embeddings_array)
        
        # Prepare visualization data
        doc_types = [m.get("type", "unknown") for m in metadatas]
        doc_ids = result["ids"]
        
        # Create DataFrame for Plotly
        plot_data = pd.DataFrame({
            "x": reduced_embeddings[:, 0],
            "y": reduced_embeddings[:, 1],
            "type": doc_types,
            "id": doc_ids
        })

        # Create interactive plot (original style)
        fig = px.scatter(
            plot_data,
            x="x",
            y="y",
            color="type",
            title=f"Document Embedding Similarity ({reduction_method} Reduced)",
            labels={"x": f"{reduction_method} 1", "y": f"{reduction_method} 2"},
            hover_name="id",
            hover_data=["type"]
        )
        
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Visualization failed: {str(e)}")

def process_zip_file(uploaded_file):
    """Process uploaded ZIP file and extract text content"""
    try:
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read()), "r") as zip_ref:
            file_list = zip_ref.namelist()
            documents = []
            
            for file_name in file_list:
                if not file_name.endswith(('.txt', '.md', '.json', '.csv', '.py', '.html')):
                    continue
                
                with zip_ref.open(file_name) as file:
                    try:
                        content = file.read().decode('utf-8')
                        documents.append({
                            "name": file_name,
                            "content": content,
                            "type": "uploaded"
                        })
                    except UnicodeDecodeError:
                        st.warning(f"Skipped binary file: {file_name}")
                        continue
    
            return documents
    
    except Exception as e:
        st.error(f"Failed to process ZIP file: {str(e)}")
        return []

def generate_synthetic_documentation_data(num_samples: int = 20) -> pd.DataFrame:
    """Generate synthetic API docs with enhanced realism and progress tracking"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        synthetic_data = []
        
        for i in range(num_samples):
            doc = {
                "endpoint": f"/api/v1/{fake.uri_path()}",
                "method": fake.random_element(["GET", "POST", "PUT", "DELETE", "PATCH"]),
                "description": fake.sentence(),
                "parameters": [{
                    "name": fake.word(),
                    "type": fake.random_element(["string", "number", "boolean", "object"]),
                    "required": fake.boolean(),
                    "description": fake.sentence()
                } for _ in range(fake.random_int(1, 5))],
                "example": {
                    "request": {fake.word(): fake.word() for _ in range(3)},
                    "response": {fake.word(): fake.word() for _ in range(5)}
                }
            }
            synthetic_data.append(doc)
            progress_bar.progress(int((i + 1) / num_samples * 100))
            status_text.text(f"Generating sample {i+1}/{num_samples}...")
        
        df = pd.DataFrame(synthetic_data)
        documents = [json.dumps(doc) for doc in synthetic_data]
        
        collection.add(
            documents=documents,
            ids=[f"synth_{i}" for i in range(len(documents))],
            metadatas=[{"type": "synthetic", "source": "auto-generated"} for _ in documents]
        )
        
        st.success("✅ Synthetic data generated!")
        return df
        
    except Exception as e:
        st.error(f"❌ Failed to generate data: {str(e)}")
        return pd.DataFrame()
    finally:
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()

def knowledge_base_controls():
    """Enhanced knowledge base management interface"""
    st.sidebar.header("🧠 Knowledge Base")
    
    # Status indicators
    with st.sidebar.container():
        doc_count = collection.count()
        st.metric("📄 Total Documents", doc_count)
    
    # Data management section
    with st.sidebar.expander("📂 Data Management", expanded=True):
        # File upload section
        st.subheader("📤 Upload Files")
        uploaded_file = st.file_uploader(
            "Upload ZIP file with documents",
            type=["zip"],
            accept_multiple_files=False,
            key="zip_uploader",
            help="Upload a ZIP file containing documentation files (TXT, MD, JSON, etc.)"
        )
        
        if uploaded_file is not None:
            if st.button("📥 Process ZIP File", key="process_zip"):
                with st.spinner("Processing ZIP file..."):
                    documents = process_zip_file(uploaded_file)
                    if documents:
                        collection.add(
                            documents=[doc["content"] for doc in documents],
                            ids=[f"upload_{i}" for i in range(len(documents))],
                            metadatas=[{"type": "uploaded", "source": doc["name"]} for doc in documents]
                        )
                        st.success(f"✅ Added {len(documents)} documents from ZIP file!")
                        time.sleep(1)
                        st.rerun()
        
        # Synthetic data generation
        st.subheader("🧪 Generate Samples")
        num_samples = st.slider("Number of samples", 1, 100, 20, key="num_samples")
        if st.button("✨ Generate Synthetic Data", key="gen_data"):
            st.session_state.synthetic_df = generate_synthetic_documentation_data(num_samples)
            st.rerun()
        
        # Clear knowledge base
        st.subheader("🛠️ Maintenance")
        if st.button("🧹 Clear Knowledge Base", key="clear_kb"):
            with st.spinner("Clearing knowledge base..."):
                collection.delete(ids=None)
                st.success("Knowledge base cleared!")
                time.sleep(1)
                st.rerun()
    
    # Show last generated data if exists
    if "synthetic_df" in st.session_state:
        with st.sidebar.expander("📊 Last Generated Data", expanded=False):
            st.dataframe(
                st.session_state.synthetic_df.head(5),
                use_container_width=True,
                hide_index=True
            )

def generate_documentation(task_type: str, input_text: str):
    """Enhanced documentation generation with RAG context"""
    if not input_text.strip():
        st.warning("Please enter content")
        return None
    
    with st.status(f"🧙‍♂️ Generating {task_type}...", expanded=True) as status:
        try:
            # Retrieve relevant context
            context = []
            if collection.count() > 0:
                results = collection.query(
                    query_texts=[input_text],
                    n_results=3
                )
                context = results["documents"][0]
            
            # Enhanced prompt engineering
            base_prompt = f"""You are DocuGenAI, a professional documentation assistant.
            Current Task: {task_type}
            Input: {input_text}"""
            
            task_guides = {
                "API Documentation": """
                Generate comprehensive API documentation including:
                1. Overview - Purpose and functionality
                2. Authentication - Methods and requirements
                3. Endpoints - Paths, methods, parameters
                4. Examples - Request/response samples
                5. Error Codes - Possible errors and solutions
                6. Rate Limits - Usage thresholds""",
                
                "Code Explanation": """
                Provide detailed code explanation covering:
                1. Purpose - What the code accomplishes
                2. Components - Key functions/classes
                3. Logic - Control flow and algorithms
                4. Edge Cases - Potential failure scenarios
                5. Improvements - Optimization suggestions""",
                
                "Code Generation": """
                Generate production-ready code including:
                1. Complete implementation
                2. Type hints and docstrings
                3. Error handling
                4. Usage examples
                5. Dependencies
                6. Tests""",
                
                "Troubleshooting Guide": """
                Create actionable troubleshooting guide with:
                1. Symptoms - How to recognize
                2. Root Causes - Likely triggers
                3. Solutions - Step-by-step fixes
                4. Prevention - Best practices
                5. Escalation - When to seek help"""
            }
            
            prompt = f"{base_prompt}\n{task_guides.get(task_type, '')}"
            
            if context:
                prompt += "\n\nRelevant Context:\n" + "\n".join([f"- {c}" for c in context])
            
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            status.update(label="✅ Documentation generated!", state="complete", expanded=False)
            return response.choices[0].message.content
            
        except Exception as e:
            status.update(label="❌ Generation failed", state="error")
            st.error(str(e))
            return None

def document_explorer():
    """Enhanced document explorer with search and filtering"""
    st.header("📚 Knowledge Base Explorer")
    
    # Search and filter controls
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input(
                "🔍 Search documents",
                placeholder="Enter keywords...",
                key="search_query"
            )
        with col2:
            doc_type_filter = st.selectbox(
                "Filter by type",
                ["All", "synthetic", "generated", "user", "uploaded"],
                key="doc_filter"
            )
    
    try:
        # Fetch and filter documents
        kb_data = collection.get()
        documents = kb_data["documents"]
        metadatas = kb_data.get("metadatas", [{}]*len(documents))
        ids = kb_data["ids"]
        
        filtered_docs = []
        for doc, meta, id in zip(documents, metadatas, ids):
            if search_query and search_query.lower() not in doc.lower():
                continue
            if doc_type_filter != "All" and meta.get("type") != doc_type_filter:
                continue
            filtered_docs.append((id, doc, meta))
        
        # Display document count
        st.caption(f"📄 Showing {len(filtered_docs)} documents")
        
        # Display documents in a nice grid
        if not filtered_docs:
            st.info("No documents match your search criteria")
        else:
            for i, (id, doc, meta) in enumerate(filtered_docs[:20]):  # Limit to 20 for performance
                with st.expander(f"📝 Document: {id}", expanded=(i < 2)):
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.markdown(f"""
                        <div style="background: #f0f2f6; padding: 12px; border-radius: 8px;">
                            <p><strong>ID:</strong> {id}</p>
                            <p><strong>Type:</strong> {meta.get('type', 'unknown')}</p>
                            <p><strong>Source:</strong> {meta.get('source', 'N/A')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button(f"🗑️ Delete", key=f"del_{id}"):
                            collection.delete(ids=[id])
                            st.rerun()
                    
                    with col2:
                        st.markdown(f"""
                        <div style="background: white; padding: 16px; border-radius: 8px; border-left: 4px solid #6e48aa;">
                            {doc[:500]}{'...' if len(doc) > 500 else ''}
                        </div>
                        """, unsafe_allow_html=True)
        
        # Visualization section
        if len(filtered_docs) > 1:
            visualize_document_similarity()
        
    except Exception as e:
        st.error(f"Knowledge base error: {str(e)}")

def documentation_generator():
    """Enhanced documentation generator interface"""
    st.header("📝 Documentation Generator")
    
    # Task selection in a nice card
    with st.container():
        st.markdown("""
        <div class="card">
            <h3>What would you like to generate?</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Task selection - modified for equal button sizes
        task_options = {
            "API Documentation": "📋",
            "Code Explanation": "🔍",
            "Code Generation": "💻",
            "Troubleshooting": "🛠️"  # Shortened label to fit better
        }
        
        # Create equal width columns
        cols = st.columns(len(task_options))
        for i, (task, emoji) in enumerate(task_options.items()):
            with cols[i]:
                # Added custom styling to ensure equal width
                st.markdown(f"""
                <style>
                    div[data-testid="stButton"]:has(button[key="task_{task}"]) {{
                        width: 100%;
                    }}
                    button[key="task_{task}"] {{
                        width: 100% !important;
                        display: flex;
                        justify-content: center;
                    }}
                </style>
                """, unsafe_allow_html=True)
                
                if st.button(f"{emoji} {task}", 
                           key=f"task_{task}", 
                           use_container_width=True):
                    st.session_state.task_type = task
        
        st.session_state.task_type = st.selectbox(
            "Select Documentation Type:",
            list(task_options.keys()),
            index=list(task_options.keys()).index(st.session_state.get("task_type", "API Documentation")),
            label_visibility="collapsed"
        )
    
    # User input in a nice card
    with st.container():
        st.markdown("""
        <div class="card">
            <h3>Input your technical content</h3>
        </div>
        """, unsafe_allow_html=True)
        
        user_input = st.text_area(
            "Enter your technical content:", 
            height=300,
            placeholder="Paste API details, code, or error description...",
            label_visibility="collapsed"
        )
        
        # Generate button with nice styling
        if st.button("🚀 Generate Documentation", type="primary", key="generate_docs", use_container_width=True):
            st.session_state.generated_docs = generate_documentation(
                st.session_state.task_type,
                user_input
            )
    
    # Display results in a nice card
    if st.session_state.get("generated_docs"):
        with st.container():
            st.markdown("""
            <div class="card">
                <h3>Generated Documentation</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(st.session_state.generated_docs)
            
            # Save to knowledge base - fixed button text
            cols = st.columns([4, 1])
            with cols[1]:
                st.markdown("""
                <style>
                    button[key="save_docs"] {{
                        white-space: nowrap;
                        overflow: hidden;
                        text-overflow: ellipsis;
                    }}
                </style>
                """, unsafe_allow_html=True)
                
                if st.button("💾 Save to KB",  # Shortened text
                           key="save_docs", 
                           help="Save to Knowledge Base",
                           use_container_width=True):
                    collection.add(
                        documents=[user_input + "\n\n" + st.session_state.generated_docs],
                        metadatas=[{"type": "generated", "task": st.session_state.task_type}],
                        ids=[f"doc_{int(time.time())}"]
                    )
                    st.success("Documentation saved!")
                    time.sleep(1)
                    st.rerun()
def main():
    # Configure page
    st.set_page_config(
        page_title="DocuGenAI",
        page_icon="💡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject custom CSS
    inject_custom_css()
    
    # Initialize session state
    if 'generated_docs' not in st.session_state:
        st.session_state.generated_docs = ""
    if 'task_type' not in st.session_state:
        st.session_state.task_type = "API Documentation"
    
    # App header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #6e48aa 0%, #9d50bb 100%); 
                padding: 20px; 
                border-radius: 10px; 
                color: white;
                margin-bottom: 20px;">
        <h1 style="margin: 0; color: white;">DocuGenAI</h1>
        <p style="margin: 0; opacity: 0.8;">Your AI-powered documentation assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main layout with tabs
    tab1, tab2 = st.tabs(["📝 Documentation Generator", "📚 Knowledge Explorer"])
    
    with tab1:
        documentation_generator()
    
    with tab2:
        document_explorer()
    
    # Sidebar controls
    knowledge_base_controls()

if __name__ == "__main__":
    main()