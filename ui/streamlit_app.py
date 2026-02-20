"""
Private AI Librarian – Streamlit UI.
Run: streamlit run ui/streamlit_app.py (from project root)
"""
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from config import UPLOADS_DIR, INDEX_DIR

st.set_page_config(
    page_title="Private AI Librarian",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Minimal custom style: clean and professional
st.markdown("""
<style>
    .stExpander { border: 1px solid #e0e0e0; border-radius: 6px; }
    .metric-card { padding: 0.5rem 0; }
    div[data-testid="stExpander"] summary { font-weight: 500; }
</style>
""", unsafe_allow_html=True)


def index_documents():
    """Trigger indexing from uploaded files."""
    from indexer import run_indexing
    result = run_indexing(source=UPLOADS_DIR, index_dir=INDEX_DIR)
    return result


def has_index():
    """Check if we have an existing index (docstore)."""
    from retrieval.docstore import docstore_path
    return docstore_path(INDEX_DIR).exists()


def main():
    st.title("📚 Private AI Librarian")
    st.caption("Fully offline · Local LLM · Hybrid retrieval")

    # Sidebar: upload and index
    with st.sidebar:
        st.header("Documents")
        uploaded = st.file_uploader(
            "Upload PDF or TXT",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )
        if uploaded:
            for f in uploaded:
                path = UPLOADS_DIR / f.name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f.getvalue())
            st.success(f"Saved {len(uploaded)} file(s). Click 'Index documents' to build the search index.")

        if st.button("Index documents", type="primary"):
            with st.spinner("Chunking and building index…"):
                result = index_documents()
            if result["status"] == "ok":
                st.success(f"Indexed {result['n_docs']} doc(s), {result['n_chunks']} chunks.")
            elif result["status"] == "no_documents":
                st.warning("No documents in upload folder. Upload files first.")
            else:
                st.info(f"Status: {result['status']} ({result.get('n_chunks', 0)} chunks)")

        st.divider()
        use_hybrid = st.radio(
            "Retrieval mode",
            ["Hybrid (dense + BM25)", "Dense only"],
            index=0,
        )
        use_hybrid = use_hybrid.startswith("Hybrid")

    # Chat
    if not has_index():
        st.info("Upload documents and click **Index documents** in the sidebar to start.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                with st.expander("Retrieved chunks"):
                    for i, c in enumerate(msg["meta"].get("retrieved_chunks", []), 1):
                        st.markdown(f"**Chunk {i}** (score: {c.get('score', 0):.3f})")
                        st.text(c.get("text", "")[:500] + ("…" if len(c.get("text", "")) > 500 else ""))
                st.caption(
                    f"Confidence: {msg['meta'].get('confidence', 0):.2f} · "
                    f"Retrieval: {msg['meta'].get('retrieval_time_s', 0):.2f}s · "
                    f"Generation: {msg['meta'].get('generation_time_s', 0):.2f}s"
                )

    if prompt := st.chat_input("Ask a question about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                from pipeline import run_pipeline
                out = run_pipeline(
                    prompt,
                    use_hybrid=use_hybrid,
                    use_query_rewrite=True,
                    index_dir=INDEX_DIR,
                )
            if out.get("error"):
                st.error(out["error"])
                meta = None
            else:
                st.markdown(out["answer"])
                # Expandable: retrieved chunks
                with st.expander("Retrieved chunks", expanded=False):
                    for i, c in enumerate(out.get("retrieved_chunks", []), 1):
                        score = c.get("score", 0)
                        text = c.get("text", "")
                        st.markdown(f"**Chunk {i}** · score: {score:.3f}")
                        st.text(text[:600] + ("…" if len(text) > 600 else ""))
                        st.divider()
                # Metrics and optional rewritten query
                if out.get("rewritten_query") and out["rewritten_query"] != prompt:
                    st.caption(f"Rewritten query: *{out['rewritten_query'][:100]}*")
                st.caption(
                    f"Confidence: **{out.get('confidence', 0):.2f}** · "
                    f"Retrieval: **{out.get('retrieval_time_s', 0):.2f}s** · "
                    f"Generation: **{out.get('generation_time_s', 0):.2f}s**"
                )
                meta = {
                    "retrieved_chunks": out.get("retrieved_chunks", []),
                    "confidence": out.get("confidence", 0),
                    "retrieval_time_s": out.get("retrieval_time_s", 0),
                    "generation_time_s": out.get("generation_time_s", 0),
                }
        st.session_state.messages.append({
            "role": "assistant",
            "content": out.get("answer", out.get("error", "")),
            "meta": meta,
        })


if __name__ == "__main__":
    main()
