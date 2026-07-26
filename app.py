import streamlit as st
from groq import Groq
from fastembed import TextEmbedding
from pypdf import PdfReader
import numpy as np

st.set_page_config(page_title="RAG Chat AI", page_icon="💬", layout="centered")
st.title("💬 RAG Chat AI")
st.caption("Upload PDFs → Chat with your documents + General AI Knowledge")

# ========== API SETUP ==========
import os

groq_key = None
groq_client = None

try:
    if "GROQ_API_KEY" in st.secrets:
        groq_key = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

if not groq_key:
    groq_key = os.getenv("GROQ_API_KEY")

if groq_key:
    try:
        groq_client = Groq(api_key=groq_key)
    except Exception as e:
        st.sidebar.error(f"Groq Error: {e}")

# ========== LOCAL EMBEDDING MODEL ==========
@st.cache_resource
def load_embedder():
    with st.spinner("Loading embedding model (22MB, one-time)..."):
        return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

embedder = None
try:
    embedder = load_embedder()
except Exception as e:
    st.sidebar.error(f"Embedder Error: {e}")

# ========== SESSION STATE ==========
if "messages" not in st.session_state:
    st.session_state.messages = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "file_stats" not in st.session_state:
    st.session_state.file_stats = []
if "ready" not in st.session_state:
    st.session_state.ready = False

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("🔌 Status")
    if groq_client:
        st.success("✅ Groq Connected")
    else:
        st.error("❌ No Groq API Key")
        st.markdown("Get free key at [console.groq.com](https://console.groq.com)")
    
    if embedder:
        st.success("✅ Local Embedder Ready")
    
    st.divider()
    
    # MODE TOGGLE
    st.subheader("🎛️ Answer Mode")
    answer_mode = st.radio(
        "Choose how the AI answers:",
        ["🧠 Hybrid (PDF + General Knowledge)", "📄 Documents Only"],
        index=0,
        help="Hybrid = uses PDFs when relevant, general knowledge otherwise. Documents Only = strictly from uploaded PDFs."
    )
    hybrid_mode = (answer_mode == "🧠 Hybrid (PDF + General Knowledge)")
    
    st.divider()
    
    # Document upload section
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type=['pdf'],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                all_chunks = []
                file_stats = []
                
                for pdf_file in uploaded_files:
                    try:
                        reader = PdfReader(pdf_file)
                        text_parts = []
                        for i, page in enumerate(reader.pages):
                            txt = page.extract_text()
                            if txt:
                                text_parts.append(f"[Page {i+1}]\n{txt}")
                        
                        full_text = "\n\n".join(text_parts)
                        pages = len(reader.pages)
                        
                        chunks = []
                        start = 0
                        while start < len(full_text):
                            end = min(start + 1000, len(full_text))
                            chunk = full_text[start:end].strip()
                            if len(chunk) > 50:
                                chunks.append(chunk)
                            start = end - 150 if end < len(full_text) else end
                        
                        all_chunks.extend(chunks)
                        file_stats.append({"name": pdf_file.name, "pages": pages, "chunks": len(chunks)})
                    except Exception as e:
                        st.error(f"Error with {pdf_file.name}: {e}")
                
                if not all_chunks:
                    st.error("No text extracted. Try text-based PDFs.")
                else:
                    embeddings = []
                    emb_progress = st.progress(0)
                    
                    for i, chunk in enumerate(all_chunks):
                        try:
                            emb_gen = embedder.embed([chunk[:8000]])
                            emb = np.array(list(emb_gen)[0], dtype=np.float32)
                            embeddings.append(emb)
                        except Exception as e:
                            st.error(f"Embed error chunk {i}: {e}")
                        emb_progress.progress((i + 1) / len(all_chunks))
                    
                    if embeddings:
                        st.session_state.embeddings = np.array(embeddings)
                        st.session_state.chunks = all_chunks
                        st.session_state.file_stats = file_stats
                        st.session_state.ready = True
                        st.success(f"✅ {len(uploaded_files)} files → {len(all_chunks)} chunks")
                    else:
                        st.error("❌ Failed to create embeddings.")
    
    if st.session_state.file_stats:
        st.divider()
        st.subheader("📊 Documents")
        for s in st.session_state.file_stats:
            st.write(f"📄 {s['name']}")
            st.caption(f"{s['pages']} pages → {s['chunks']} chunks")
    
    if st.session_state.messages:
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    st.divider()
    st.markdown("""
    ### ⏱️ Free Tier
    - **Groq**: 30 req/min
    - **Embeddings**: Unlimited (local)
    """)

# ========== SETUP CHECK ==========
if not groq_client or not embedder:
    st.warning("⚠️ Setup Required")
    st.markdown("""
    ### Step 1: Get Groq API Key (Free, No Credit Card)
    1. Go to https://console.groq.com
    2. Sign up → API Keys → Create Key
    
    ### Step 2: Add to Streamlit Cloud Secrets
    `GROQ_API_KEY = "your-key"`
    
    ### Step 3: Upload PDFs in the sidebar
    """)
    st.stop()

if not st.session_state.ready:
    st.info("👈 **Upload PDF files in the sidebar and click 'Process Documents' to start chatting.**")
    st.markdown("""
    ### 💡 How it works:
    1. Upload your PDF documents in the sidebar
    2. Click **Process Documents**
    3. Ask questions — the AI uses your PDFs when relevant, and general knowledge for everything else!
    """)
    st.stop()

# ========== CHAT INTERFACE ==========
st.markdown("---")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show source badge for assistant messages
        if message["role"] == "assistant":
            if message.get("source_type") == "document":
                st.caption("📄 Answered from uploaded documents")
            elif message.get("source_type") == "general":
                st.caption("🧠 Answered from general knowledge")
            
            if "sources" in message and message["sources"]:
                with st.expander("📄 View source chunks"):
                    for src in message["sources"]:
                        st.markdown(f"**Chunk** (score: {src['score']:.3f})")
                        st.text(src["text"][:600])
                        st.divider()

# Chat input at the bottom
if prompt := st.chat_input("Ask anything about your documents... or anything else!"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # 1. Embed question
                q_emb_gen = embedder.embed([prompt[:8000]])
                q_vec = np.array(list(q_emb_gen)[0], dtype=np.float32)
                
                # 2. Find top 3 similar chunks
                sims = []
                for emb in st.session_state.embeddings:
                    sim = np.dot(q_vec, emb) / (np.linalg.norm(q_vec) * np.linalg.norm(emb))
                    sims.append(sim)
                
                top_idx = np.argsort(sims)[-3:][::-1]
                top_scores = [sims[i] for i in top_idx]
                
                # 3. Check if documents are actually relevant
                best_score = top_scores[0] if top_scores else 0
                is_relevant = best_score > 0.55  # Threshold for document relevance
                
                # 4. Build context
                relevant_chunks = [st.session_state.chunks[i] for i in top_idx]
                context = "\n\n---\n\n".join(relevant_chunks)
                
                # 5. Build chat-aware prompt based on mode
                recent_history = ""
                if len(st.session_state.messages) > 2:
                    recent = st.session_state.messages[-6:-1]
                    recent_history = "\n\n".join([
                        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                        for m in recent
                    ])
                
                if hybrid_mode:
                    # HYBRID MODE: Use docs when relevant, general knowledge otherwise
                    if is_relevant:
                        system_prompt = f"""You are a helpful assistant. The user has uploaded documents that may contain relevant information. 
Use the document context below to answer if it helps. If the documents don't fully answer the question, supplement with your general knowledge.

=== RELEVANT DOCUMENT CONTEXT ===
{context}

=== RECENT CONVERSATION ===
{recent_history}

=== USER QUESTION ===
{prompt}

=== YOUR ANSWER ===
Provide a clear, accurate, and helpful answer. When using information from the documents, be precise."""
                        source_type = "document"
                    else:
                        system_prompt = f"""You are a helpful assistant. The user asked a question that doesn't seem related to their uploaded documents. 
Answer using your general knowledge. Be helpful and accurate.

=== RECENT CONVERSATION ===
{recent_history}

=== USER QUESTION ===
{prompt}

=== YOUR ANSWER ===
Provide a clear, accurate, and helpful answer."""
                        source_type = "general"
                else:
                    # STRICT MODE: Only from documents
                    system_prompt = f"""You are a helpful study assistant. Answer the user's question using ONLY the information provided in the context below.
If the answer is not found in the context, say: "I don't have enough information in the uploaded documents to answer this."

=== CONTEXT FROM DOCUMENTS ===
{context}

=== RECENT CONVERSATION ===
{recent_history}

=== USER QUESTION ===
{prompt}

=== YOUR ANSWER ===
Provide a clear, accurate, and concise answer."""
                    source_type = "document" if is_relevant else "general"
                
                # 6. Call Groq
                chat_completion = groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": system_prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                    max_tokens=1024
                )
                
                answer = chat_completion.choices[0].message.content
                
                # 7. Prepare sources (only if documents were relevant)
                sources = []
                if is_relevant:
                    for i, idx in enumerate(top_idx):
                        sources.append({
                            "text": st.session_state.chunks[idx],
                            "score": sims[idx]
                        })
                
                # 8. Display answer
                st.markdown(answer)
                
                if is_relevant and sources:
                    st.caption("📄 Answered from uploaded documents")
                    with st.expander("📄 View source chunks"):
                        for src in sources:
                            st.markdown(f"**Chunk** (score: {src['score']:.3f})")
                            st.text(src["text"][:600])
                            st.divider()
                else:
                    st.caption("🧠 Answered from general knowledge")
                
                # 9. Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources if is_relevant else [],
                    "source_type": source_type
                })
            
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg:
                    st.error("⏳ Rate limit (30/min). Wait a few seconds and try again.")
                else:
                    st.error(f"Error: {e}")
