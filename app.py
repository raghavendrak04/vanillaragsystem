"""
Defence RAG System — Streamlit Frontend

A clean, minimalistic, transparent UI for querying Indian defence
procurement policy, financial delegations, and naval regulations.
"""

import sys
import os
import time
from pathlib import Path

# Fix encoding
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# Must be the FIRST Streamlit command
st.set_page_config(
    page_title="Defence RAG System",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

import config
from ingest.loader import load_documents
from ingest.chunker import chunk_documents, Chunk
from ingest.embedder import embed_texts
from retriever.vector_store import (
    add_chunks, reset_collection, get_collection_stats, get_collection
)
from retriever.search import search, SearchResult
from generator.llm import generate_answer


# ─────────────────────────────────────────────────────────────
# Custom CSS — Clean, Minimalistic, Transparent Design
# ─────────────────────────────────────────────────────────────
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global Reset ── */
    *, *::before, *::after { box-sizing: border-box; }

    .stApp {
        background: linear-gradient(135deg, #0a0e17 0%, #0d1321 30%, #111827 60%, #0f172a 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ── Remove default streamlit padding ── */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1100px !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.85) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.15) !important;
    }

    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] label {
        color: #94a3b8 !important;
        font-size: 0.875rem !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #e2e8f0 !important;
    }

    /* ── Typography ── */
    h1, h2, h3 {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }

    p, li, span, label {
        color: #cbd5e1 !important;
    }

    /* ── Glass Card ── */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(99, 102, 241, 0.12);
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.08);
    }

    /* ── Hero Header ── */
    .hero-header {
        text-align: center;
        padding: 2rem 0 1.5rem 0;
    }

    .hero-header h1 {
        font-size: 2.25rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #818cf8, #6366f1, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem !important;
    }

    .hero-subtitle {
        color: #64748b !important;
        font-size: 1rem;
        font-weight: 400;
        letter-spacing: 0.02em;
    }

    /* ── Stats Row ── */
    .stat-card {
        background: rgba(30, 41, 59, 0.35);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(99, 102, 241, 0.1);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }

    .stat-value {
        font-size: 1.75rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818cf8, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .stat-label {
        font-size: 0.75rem;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.25rem;
    }

    /* ── Question Input ── */
    .stTextArea textarea,
    .stTextInput input {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 0.85rem 1rem !important;
        transition: all 0.3s ease !important;
    }

    .stTextArea textarea:focus,
    .stTextInput input:focus {
        border-color: rgba(99, 102, 241, 0.5) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.75rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.25) !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 24px rgba(99, 102, 241, 0.35) !important;
    }

    /* ── Answer Box ── */
    .answer-box {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-left: 3px solid #6366f1;
        border-radius: 12px;
        padding: 1.5rem 1.75rem;
        margin: 1rem 0;
        line-height: 1.7;
    }

    .answer-box p, .answer-box li {
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
    }

    /* ── Citation Chip ── */
    .citation-chip {
        display: inline-block;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 8px;
        padding: 0.4rem 0.85rem;
        margin: 0.25rem 0.35rem 0.25rem 0;
        font-size: 0.78rem;
        color: #a5b4fc !important;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .citation-chip:hover {
        background: rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.35);
    }

    /* ── Context Chunk ── */
    .context-chunk {
        background: rgba(30, 41, 59, 0.3);
        border: 1px solid rgba(99, 102, 241, 0.08);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        transition: all 0.2s ease;
    }

    .context-chunk:hover {
        border-color: rgba(99, 102, 241, 0.2);
    }

    .chunk-header {
        font-size: 0.75rem;
        color: #818cf8 !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.5rem;
    }

    .chunk-text {
        font-size: 0.85rem;
        color: #94a3b8 !important;
        line-height: 1.6;
    }

    /* ── Score Bar ── */
    .score-bar-container {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 0.25rem 0;
    }

    .score-bar {
        flex: 1;
        height: 4px;
        background: rgba(99, 102, 241, 0.1);
        border-radius: 4px;
        overflow: hidden;
    }

    .score-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }

    .score-value {
        font-size: 0.75rem;
        color: #818cf8 !important;
        font-weight: 600;
        min-width: 40px;
        text-align: right;
    }

    /* ── Divider ── */
    .subtle-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.15), transparent);
        margin: 1.5rem 0;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.3) !important;
        border-radius: 10px !important;
        color: #94a3b8 !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.3);
        border: 1px solid rgba(99, 102, 241, 0.08);
        border-radius: 12px;
        padding: 1rem !important;
    }

    [data-testid="stMetricValue"] {
        color: #818cf8 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #64748b !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #6366f1 !important;
    }

    /* ── Selectbox / Slider ── */
    .stSelectbox > div > div,
    .stSlider > div {
        color: #e2e8f0 !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border: 1px solid rgba(99, 102, 241, 0.15) !important;
        border-radius: 8px !important;
        color: #94a3b8 !important;
        padding: 0.5rem 1rem !important;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.15) !important;
        border-color: rgba(99, 102, 241, 0.3) !important;
        color: #a5b4fc !important;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Animations ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .animate-in {
        animation: fadeInUp 0.4s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# UI Components
# ─────────────────────────────────────────────────────────────

def render_hero():
    """Render the hero header."""
    st.markdown("""
    <div class="hero-header animate-in">
        <h1>Defence RAG System</h1>
        <p class="hero-subtitle">
            Intelligent Question Answering over Indian Naval Regulations
        </p>
    </div>
    <hr class="subtle-divider">
    """, unsafe_allow_html=True)


def render_stats():
    """Render index statistics."""
    stats = get_collection_stats()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{stats['count']}</div>
            <div class="stat-label">Indexed Chunks</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        docs = stats.get("sample_docs", [])
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(docs)}</div>
            <div class="stat-label">Documents</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{config.TOP_K}</div>
            <div class="stat-label">Top-K Retrieval</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        model_short = config.get_llm_model().split("/")[-1] if "/" in config.get_llm_model() else config.get_llm_model()
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="font-size: 1rem;">{model_short}</div>
            <div class="stat-label">LLM Model</div>
        </div>
        """, unsafe_allow_html=True)


def render_answer(response: dict, elapsed: float):
    """Render the generated answer with citations."""
    answer = response["answer"]

    # Answer section
    st.markdown("### Answer")
    st.markdown(f"""
    <div class="answer-box animate-in">
        {_markdown_to_html(answer)}
    </div>
    """, unsafe_allow_html=True)

    # Metadata row
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric("Total Time", f"{elapsed:.1f}s")
    with mcol2:
        st.metric("Search & Fetch", f"{response.get('search_time', 0):.2f}s")
    with mcol3:
        st.metric("LLM Generation", f"{response.get('gen_time', 0):.2f}s")
    with mcol4:
        st.metric("Model Used", response["model"].split("/")[-1])

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # Citations
    if response.get("citations"):
        st.markdown("### Sources")
        chips_html = ""
        for citation in response["citations"]:
            chips_html += f'<span class="citation-chip">{citation}</span>'
        st.markdown(f'<div class="animate-in">{chips_html}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # Retrieved context chunks
    if response.get("search_results"):
        with st.expander(f"Retrieved Context ({len(response['search_results'])} chunks)", expanded=False):
            for r in response["search_results"]:
                score = r.similarity_score
                color = _score_color(score)
                st.markdown(f"""
                <div class="context-chunk">
                    <div class="chunk-header">
                        #{r.rank} &mdash; {r.doc_name}, Page {r.page_number}
                        {f' &mdash; {r.section_title}' if r.section_title else ''}
                    </div>
                    <div class="score-bar-container">
                        <div class="score-bar">
                            <div class="score-bar-fill" style="width: {score*100:.0f}%; background: {color};"></div>
                        </div>
                        <div class="score-value">{score:.2f}</div>
                    </div>
                    <div class="chunk-text">{r.text[:400]}{'...' if len(r.text) > 400 else ''}</div>
                </div>
                """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with controls."""
    with st.sidebar:
        st.markdown("## Settings")

        st.markdown("---")

        # Provider info
        st.markdown(f"**Embedding:** `{config.get_embedding_model().split('/')[-1]}`")
        
        selected_provider = st.selectbox(
            "LLM Provider",
            options=["gemini", "openai"],
            index=0 if config.LLM_PROVIDER == "gemini" else 1,
            help="Select which AI model to use for generating answers. Note: OpenAI requires a valid key with credits."
        )

        st.markdown("---")

        # Retrieval settings
        top_k = st.slider("Top-K Chunks", min_value=1, max_value=15, value=config.TOP_K, key="top_k")

        st.markdown("---")

        # Index management
        st.markdown("## Index")
        stats = get_collection_stats()
        st.markdown(f"**Chunks indexed:** {stats['count']}")
        if stats.get("sample_docs"):
            st.markdown(f"**Documents:** {', '.join(stats['sample_docs'])}")

        if stats["count"] == 0:
            if st.button("Index Documents", key="btn_ingest"):
                _run_ingest()

        if stats["count"] > 0:
            if st.button("Re-index Documents", key="btn_reingest"):
                _run_ingest()

        st.markdown("---")

        # Sample questions
        st.markdown("## Sample Questions")
        sample_questions = [
            "What are the eligibility criteria for recruitment?",
            "What is the definition of Emergency?",
            "Who is the Administrative Authority?",
            "What is the tenure of appointment for permanent staff?",
            "Who constitutes the Service?",
        ]
        for q in sample_questions:
            if st.button(q, key=f"sample_{hash(q)}", use_container_width=True):
                st.session_state["question_input"] = q
                st.rerun()

    return top_k, selected_provider


def _run_ingest():
    """Run the ingest pipeline with progress."""
    with st.spinner("Loading and indexing documents..."):
        try:
            # Load
            pages = load_documents(config.SOURCE_DIR)
            if not pages:
                st.error("No pages extracted from PDFs.")
                return

            # Chunk
            chunks = chunk_documents(pages)
            if not chunks:
                st.error("No chunks created.")
                return

            # Reset existing
            reset_collection()

            # Embed
            texts = [c.text for c in chunks]
            embeddings = embed_texts(texts)

            # Store
            add_chunks(chunks, embeddings)

            st.success(f"Indexed {len(chunks)} chunks from {len(set(p.metadata['doc_name'] for p in pages))} document(s)")
            time.sleep(1)
            st.rerun()

        except Exception as e:
            st.error(f"Ingest failed: {e}")


def _markdown_to_html(text: str) -> str:
    """Basic markdown to HTML for the answer box."""
    import re

    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    # Bullet points
    lines = text.split('\n')
    html_lines = []
    in_list = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('- ') or stripped.startswith('* '):
            if not in_list:
                html_lines.append('<ul style="margin: 0.5rem 0; padding-left: 1.5rem;">')
                in_list = True
            html_lines.append(f'<li style="margin-bottom: 0.3rem;">{stripped[2:]}</li>')
        elif stripped.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
            if not in_list:
                html_lines.append('<ol style="margin: 0.5rem 0; padding-left: 1.5rem;">')
                in_list = True
            content = re.sub(r'^\d+\.\s*', '', stripped)
            html_lines.append(f'<li style="margin-bottom: 0.3rem;">{content}</li>')
        else:
            if in_list:
                html_lines.append('</ul>' if html_lines[-2].startswith('<ul') or '<li>' in html_lines[-1] else '</ol>')
                in_list = False
            if stripped:
                html_lines.append(f'<p style="margin: 0.5rem 0;">{stripped}</p>')
    if in_list:
        html_lines.append('</ul>')

    return '\n'.join(html_lines)


def _score_color(score: float) -> str:
    """Get color for relevance score."""
    if score >= 0.8:
        return "linear-gradient(90deg, #22c55e, #4ade80)"
    elif score >= 0.6:
        return "linear-gradient(90deg, #6366f1, #818cf8)"
    elif score >= 0.4:
        return "linear-gradient(90deg, #f59e0b, #fbbf24)"
    else:
        return "linear-gradient(90deg, #ef4444, #f87171)"


# ─────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────

def main():
    inject_custom_css()
    render_hero()

    # Sidebar
    top_k, llm_provider = render_sidebar()

    # Stats bar
    stats = get_collection_stats()
    if stats["count"] > 0:
        render_stats()
        st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 3rem;">
            <h3 style="color: #818cf8 !important;">No Documents Indexed</h3>
            <p style="color: #64748b !important;">
                Click <strong>Index Documents</strong> in the sidebar to get started.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Query input
    st.markdown("### Ask a Question")

    # Use session state for sample question injection
    default_q = st.session_state.get("question_input", "")
    question = st.text_area(
        "Enter your question about Indian Naval Regulations",
        value=default_q,
        height=80,
        placeholder="e.g., What are the eligibility criteria for recruitment to the Indian Naval Auxiliary Service?",
        label_visibility="collapsed",
        key="query_box",
    )

    col_btn, col_space = st.columns([1, 4])
    with col_btn:
        ask_clicked = st.button("Search & Answer", type="primary", use_container_width=True)

    # Clear the injected question after it's been used
    if "question_input" in st.session_state:
        del st.session_state["question_input"]

    # Process query
    if ask_clicked and question.strip():
        st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

        start = time.time()

        try:
            with st.status("Processing your query...", expanded=True) as status:
                # Search
                status.write("🔍 Searching vector database for relevant context...")
                t0 = time.time()
                results = search(question.strip(), top_k=top_k)
                search_time = time.time() - t0
                status.write(f"✅ Found {len(results)} relevant chunks in **{search_time:.2f}s**.")

                # Generate
                status.write(f"🧠 Generating answer using `{llm_provider}`...")
                t1 = time.time()
                response = generate_answer(question.strip(), results, provider=llm_provider)
                gen_time = time.time() - t1
                status.write(f"✅ Answer generated in **{gen_time:.2f}s**.")

                elapsed = time.time() - start
                status.update(label=f"Query completed in {elapsed:.2f}s", state="complete", expanded=False)

            # Render
            response["search_time"] = search_time
            response["gen_time"] = gen_time
            render_answer(response, elapsed)

            # Store in history
            if "history" not in st.session_state:
                st.session_state["history"] = []
            st.session_state["history"].insert(0, {
                "question": question.strip(),
                "answer": response["answer"],
                "time": elapsed,
                "citations": response.get("citations", []),
            })

        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Query history
    if st.session_state.get("history"):
        st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)
        st.markdown("### Recent Queries")
        for i, h in enumerate(st.session_state["history"][:5]):
            with st.expander(f"{h['question'][:80]}... ({h['time']:.1f}s)" if len(h['question']) > 80 else f"{h['question']} ({h['time']:.1f}s)"):
                st.markdown(h["answer"])
                if h.get("citations"):
                    chips = " ".join([f'`{c}`' for c in h["citations"][:3]])
                    st.markdown(f"**Sources:** {chips}")


if __name__ == "__main__":
    main()
