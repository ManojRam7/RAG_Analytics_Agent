"""Streamlit UI for the RAG Analytics Agent.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import pathlib
import sys

import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

from rag_assistant import vectorstore as vs_mod  # noqa: E402
from rag_assistant.pipeline import RAGPipeline  # noqa: E402

st.set_page_config(page_title="RAG Analytics Agent", layout="wide")

SAMPLE_QUESTIONS = [
    "What drove the revenue growth in Q3?",
    "Which department had the highest attrition?",
    "Which marketing channel had the best ROI?",
    "Which sales region is growing fastest?",
    "What is the biggest supply chain risk?",
]


@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def _reports_present(rag: RAGPipeline) -> bool:
    d = rag.cfg.reports_path
    return d.exists() and any(
        p.suffix.lower() in {".txt", ".md", ".pdf", ".docx"} for p in d.rglob("*")
    )


def main() -> None:
    st.title("RAG Analytics Agent for Business Reports")
    st.caption(
        "Ask questions about your business reports and get **source-grounded** answers "
        "with inline [S#] citations. Built with LangChain + FAISS."
    )

    rag = get_pipeline()

    # ---------------------------------------------------------------- sidebar
    with st.sidebar:
        st.header("Status")
        st.markdown(f"**LLM backend:** `{rag.generator.name}`")
        st.markdown(f"**Embeddings:** `{rag.embeddings_name}`")
        index_ready = vs_mod.index_exists(rag.cfg.index_path)
        st.markdown(f"**Index:** {'ready' if index_ready else 'not built yet'}")

        st.divider()
        k = st.slider("Chunks to retrieve (k)", 1, 10, rag.cfg.retrieval.k)

        st.divider()
        st.subheader("Manage corpus")
        if not _reports_present(rag):
            st.info("No reports found. Generate the sample set to get started.")
            if st.button("Generate sample reports"):
                import scripts.generate_sample_reports as gen  # type: ignore

                gen.main()
                st.success("Sample reports created. Now build the index.")

        uploaded = st.file_uploader(
            "Add your own reports",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
        )
        if uploaded:
            rag.cfg.reports_path.mkdir(parents=True, exist_ok=True)
            for f in uploaded:
                (rag.cfg.reports_path / f.name).write_bytes(f.getbuffer())
            st.success(f"Saved {len(uploaded)} file(s). Rebuild the index to include them.")

        if st.button("Build / rebuild index", type="primary"):
            with st.spinner("Indexing reports…"):
                try:
                    stats = rag.build_index()
                    st.success(
                        f"Indexed {stats['documents']} docs → {stats['chunks']} chunks."
                    )
                except FileNotFoundError as exc:
                    st.error(str(exc))

        st.divider()
        st.caption(
            "For better answers set `GOOGLE_API_KEY` (Gemini) or `OPENAI_API_KEY` in `.env`, "
            "or install `requirements-llm.txt` for local models. "
            "The active backend is shown above."
        )

    # ------------------------------------------------------------------- main
    if not vs_mod.index_exists(rag.cfg.index_path):
        st.warning(
            "No index yet. Use **Build / rebuild index** in the sidebar "
            "(generate the sample reports first if needed)."
        )
        return

    st.write("**Try one of these:**")
    cols = st.columns(len(SAMPLE_QUESTIONS))
    clicked = None
    for col, q in zip(cols, SAMPLE_QUESTIONS):
        if col.button(q, use_container_width=True):
            clicked = q

    question = st.text_input(
        "Your question", value=clicked or "", placeholder="e.g. How did gross margin change?"
    )
    ask = st.button("Ask", type="primary") or bool(clicked)

    if ask and question.strip():
        with st.spinner("Retrieving and answering…"):
            result = rag.query(question, k=k)

        st.subheader("Answer")
        st.markdown(result["answer"])

        st.subheader("Sources")
        for s in result["sources"]:
            page = f" · p.{s['page'] + 1}" if isinstance(s.get("page"), int) else ""
            with st.expander(f"[{s['tag']}] {s['source']}{page}  ·  score {s['score']:.3f}"):
                st.write(s["snippet"])

        st.caption(
            f"backend: llm=`{result['backend']['llm']}` · "
            f"embeddings=`{result['backend']['embeddings']}`"
        )


if __name__ == "__main__":
    main()
