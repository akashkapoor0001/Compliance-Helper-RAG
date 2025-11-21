# app.py
import streamlit as st
import os
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
from models.llm import get_chatgroq_model
from utils.response_formatter import build_system_prompt
from utils.retriever import load_index_and_meta, retrieve
from utils.ingest import index_documents
from config import config


# -------------------------------------------------
# Function to call LLM with system + conversation
# -------------------------------------------------
def get_chat_response(chat_model, messages, system_prompt):
    try:
        formatted = [SystemMessage(content=system_prompt)]

        for msg in messages:
            if msg["role"] == "user":
                formatted.append(HumanMessage(content=msg["content"]))
            else:
                formatted.append(AIMessage(content=msg["content"]))

        response = chat_model.invoke(formatted)
        text = response.content

        # Optional cleanup
        # Remove accidental repeated newlines, parentheses, messy chunks
        import re
        text = re.sub(r"\s{3,}", "\n\n", text)
        text = re.sub(r"\((Snippet ID.*?)\)", "", text)

        return text

    except Exception as e:
        return f"Error generating response: {str(e)}"


# -------------------------------------------------
# Instructions Page
# -------------------------------------------------
def instructions_page():
    st.title("Compliance Helper — Setup Instructions")

    st.markdown("""
    ### 📘 How to use this Assistant

    This Compliance Helper uses:
    - **Local HuggingFace embeddings** (fast + free)
    - **Groq LLaMA 3.1 models** for question-answering (fastest in class)

    #### Steps:
    1. Upload policy PDF/TXT files from the sidebar.
    2. Click **Build index**.
    3. Ask any compliance question.
    4. The assistant answers strictly using retrieved policy snippets.
    5. All answers include **correct structured citations** like:  
       `[Terms.pdf#12]`
    """)

    st.markdown("---")


# -------------------------------------------------
# Main Chat Page
# -------------------------------------------------
def chat_page():
    st.title("🤖 Compliance Helper — Policy RAG Assistant")

    # Initialize LLM
    try:
        chat_model = get_chatgroq_model()
    except Exception as e:
        chat_model = None
        st.error(f"LLM initialization failed: {e}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar
    with st.sidebar:
        st.header("📄 Document Indexing")

        uploaded = st.file_uploader(
            "Upload PDF/TXT files",
            accept_multiple_files=True,
            type=["pdf", "txt"]
        )

        if st.button("Build index"):
            if not uploaded:
                st.error("Please upload at least one document.")
            else:
                tmp_dir = Path("data/uploaded")
                tmp_dir.mkdir(parents=True, exist_ok=True)

                file_paths = []
                for f in uploaded:
                    out_path = tmp_dir / f.name
                    with open(out_path, "wb") as fh:
                        fh.write(f.read())
                    file_paths.append(str(out_path))

                try:
                    with st.spinner("Indexing... Please wait."):
                        _, meta = index_documents(file_paths, save_index=True)
                    st.success(f"Indexed {len(meta)} chunks from {len(file_paths)} files!")
                except Exception as e:
                    st.error(f"Indexing failed: {str(e)}")

        st.markdown("---")
        st.header("⚙️ Settings")

        mode = st.radio("Response mode", ["concise", "detailed"], index=0)

        max_k = st.slider(
            "Snippets to retrieve (k)",
            min_value=1,
            max_value=6,
            value=3,
        )

        st.markdown("---")

        if st.button("Clear chat"):
            st.session_state.messages = []
            return

    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input("Ask your compliance question...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Make sure index exists
        try:
            index, metadata = load_index_and_meta()
        except Exception as e:
            st.error(f"Index not found — please upload documents and build index first. ({e})")
            return

        # Retrieval
        with st.spinner("Retrieving relevant policy snippets..."):
            try:
                retrieved = retrieve(prompt, index, metadata, k=max_k)
            except Exception as e:
                st.error(f"Retrieval error: {str(e)}")
                retrieved = []

        # Build the strict system prompt
        system_prompt = build_system_prompt(retrieved)

        if mode == "concise":
            system_prompt += "\nRespond concisely."
        else:
            system_prompt += "\nProvide a detailed explanation."

        # LLM Response
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                if not chat_model:
                    st.error("LLM not available. Check GROQ_API_KEY and model setup.")
                    return

                response_text = get_chat_response(chat_model, st.session_state.messages, system_prompt)
                st.markdown(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})

        # Show retrieved snippets
        st.markdown("---")
        st.subheader("📌 Retrieved Policy Snippets")

        if not retrieved:
            st.write("No snippets retrieved.")
        else:
            for r in retrieved:
                st.markdown(f"**{r['doc_id']}#{r['chunk_id']}** — similarity: {r['score']:.3f}")
                snippet = r["text"]
                st.write(snippet[:1000] + ("..." if len(snippet) > 1000 else ""))


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    st.set_page_config(
        page_title="Compliance Helper",
        page_icon="🤖",
        layout="wide"
    )

    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to:", ["Chat", "Instructions"], index=0)

    if page == "Instructions":
        instructions_page()
    else:
        chat_page()


if __name__ == "__main__":
    main()
