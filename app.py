import json
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

from backend.btw_handler import handle_btw
from backend.config import (
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_LLM_MODELS,
    get_embedding_model,
    get_llm_model,
    load_user_profile,
    read_env,
    save_user_profile,
    write_env,
)
from backend.paper_loader import load_arxiv, load_document, load_webpage
from backend.rag_graph import build_graph, get_llm
from backend.vector_store import add_paper, list_papers

st.set_page_config(page_title="Papeer — Research Assistant", page_icon="📚", layout="centered")


@st.cache_resource
def get_graph():
    return build_graph()


SESSIONS_FILE = Path("sessions.json")


def load_sessions() -> dict:
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_sessions(sessions_meta: dict) -> None:
    SESSIONS_FILE.write_text(json.dumps(sessions_meta, indent=2), encoding="utf-8")


def _serialize_state(values: dict) -> dict:
    out = {}
    for k, v in values.items():
        if k == "messages":
            out[k] = [
                {
                    "type": type(m).__name__,
                    "content": (
                        m.content[:300]
                        if isinstance(m.content, str)
                        else repr(m.content)[:300]
                    ),
                }
                for m in (v or [])
            ]
        elif k == "retrieved_docs":
            out[k] = [
                {"content": d.page_content[:300], "metadata": d.metadata}
                for d in (v or [])
            ]
        else:
            out[k] = v
    return out


def generate_session_name(first_message: str) -> str:
    try:
        llm = get_llm()
        response = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Generate a concise 3-5 word title for a research chat session "
                        "based on the user's first message. Return only the title, "
                        "no punctuation at the end, no quotes."
                    ),
                },
                {"role": "user", "content": first_message[:500]},
            ]
        )
        return response.content.strip()
    except Exception:
        return "New Session"


def maybe_rename_session(session_id: str, first_message: str) -> None:
    if st.session_state.sessions_meta.get(session_id, {}).get("is_named"):
        return
    name = generate_session_name(first_message)
    st.session_state.sessions_meta[session_id]["name"] = name
    st.session_state.sessions_meta[session_id]["is_named"] = True
    save_sessions(st.session_state.sessions_meta)


def create_session() -> str:
    sid = str(uuid.uuid4())
    st.session_state.sessions_meta[sid] = {
        "id": sid,
        "name": "New Session",
        "created_at": datetime.now().isoformat(),
        "is_named": False,
    }
    save_sessions(st.session_state.sessions_meta)
    st.session_state.chats[sid] = []
    st.session_state.turns[sid] = 0
    return sid


def load_session_chats(session_id: str) -> list[dict]:
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        if not state or not state.values:
            return []
        chats = []
        turn = 0
        for msg in state.values.get("messages", []):
            type_name = type(msg).__name__
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if type_name == "HumanMessage":
                chats.append({"role": "user", "content": content})
            elif type_name in ("AIMessage", "AIMessageChunk"):
                turn += 1
                chats.append({"role": "assistant", "content": content, "turn": turn, "graph_state": {}})
        return chats
    except Exception:
        return []


def switch_session(session_id: str) -> None:
    st.session_state.active_session_id = session_id
    if session_id not in st.session_state.chats:
        st.session_state.chats[session_id] = load_session_chats(session_id)
    if session_id not in st.session_state.turns:
        turn_count = sum(1 for m in st.session_state.chats[session_id] if m["role"] == "assistant")
        st.session_state.turns[session_id] = turn_count


graph = get_graph()

# ── Bootstrap State ────────────────────────────────────────────────────────────
if "sessions_meta" not in st.session_state:
    st.session_state.sessions_meta = load_sessions()
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "turns" not in st.session_state:
    st.session_state.turns = {}
if "active_session_id" not in st.session_state:
    if st.session_state.sessions_meta:
        latest = max(
            st.session_state.sessions_meta.values(),
            key=lambda s: s["created_at"],
        )
        switch_session(latest["id"])
    else:
        sid = create_session()
        st.session_state.active_session_id = sid

if "user_profile" not in st.session_state:
    st.session_state.user_profile = load_user_profile()

active_sid = st.session_state.active_session_id
env_dict = read_env()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    profile = st.session_state.user_profile
    st.markdown(
        f"### {profile.get('avatar', '🎓')} {profile.get('name', 'Researcher')}\n"
        f"*{profile.get('role', 'Academic Researcher')}*"
    )

    # ── User Profile & API Settings Expander ───────────────────────────────────
    with st.expander("⚙️ Profile & API Settings", expanded=False):
        st.markdown("#### 👤 User Profile")
        col1, col2 = st.columns([1, 3])
        with col1:
            new_avatar = st.selectbox(
                "Avatar",
                ["🎓", "🔬", "🧠", "📚", "🤖", "💻", "💡", "🚀"],
                index=["🎓", "🔬", "🧠", "📚", "🤖", "💻", "💡", "🚀"].index(profile.get("avatar", "🎓"))
                if profile.get("avatar", "🎓") in ["🎓", "🔬", "🧠", "📚", "🤖", "💻", "💡", "🚀"]
                else 0,
            )
        with col2:
            new_name = st.text_input("Name", value=profile.get("name", "Researcher"))

        new_role = st.text_input("Role / Department", value=profile.get("role", "Academic Researcher"))
        new_focus = st.text_input("Research Focus", value=profile.get("focus_area", "Machine Learning & AI"))

        st.divider()
        st.markdown("#### 🔑 Model Selection & API Keys")
        st.caption("Provide your own API keys to directly control costs.")

        current_llm = env_dict.get("OPENAI_MODEL", get_llm_model())
        llm_idx = SUPPORTED_LLM_MODELS.index(current_llm) if current_llm in SUPPORTED_LLM_MODELS else 0
        selected_llm = st.selectbox(
            "🧠 LLM Model",
            options=SUPPORTED_LLM_MODELS,
            index=llm_idx,
            help="Select the OpenAI LLM used for routing, synthesis, and fact-checking.",
        )

        current_embed = env_dict.get("EMBEDDING_MODEL", get_embedding_model())
        embed_keys = list(SUPPORTED_EMBEDDING_MODELS.keys())
        embed_idx = embed_keys.index(current_embed) if current_embed in embed_keys else 0
        selected_embed = st.selectbox(
            "📐 Embedding Model",
            options=embed_keys,
            index=embed_idx,
            help="Select the embedding model used for vector representations.",
        )

        openai_key_input = st.text_input(
            "OpenAI API Key",
            value=env_dict.get("OPENAI_API_KEY", ""),
            type="password",
            placeholder="sk-...",
        )
        tavily_key_input = st.text_input(
            "Tavily API Key (for web & claims)",
            value=env_dict.get("TAVILY_API_KEY", ""),
            type="password",
            placeholder="tvly-...",
        )
        qdrant_url_input = st.text_input(
            "Qdrant URL",
            value=env_dict.get("QDRANT_URL", ""),
            placeholder="https://...cloud.qdrant.io",
        )
        qdrant_key_input = st.text_input(
            "Qdrant API Key",
            value=env_dict.get("QDRANT_API_KEY", ""),
            type="password",
            placeholder="ey...",
        )

        if st.button("💾 Save Profile & Keys", use_container_width=True, type="primary"):
            # Update Profile
            updated_profile = {
                "name": new_name.strip() or "Researcher",
                "avatar": new_avatar,
                "role": new_role.strip(),
                "focus_area": new_focus.strip(),
            }
            save_user_profile(updated_profile)
            st.session_state.user_profile = updated_profile

            # Update .env
            env_updates = {
                "OPENAI_API_KEY": openai_key_input.strip(),
                "OPENAI_MODEL": selected_llm,
                "EMBEDDING_MODEL": selected_embed,
                "TAVILY_API_KEY": tavily_key_input.strip(),
                "QDRANT_URL": qdrant_url_input.strip(),
                "QDRANT_API_KEY": qdrant_key_input.strip(),
            }
            write_env(env_updates)
            st.cache_resource.clear()
            st.success("✅ Profile and settings saved successfully!")
            st.rerun()

    st.divider()

    if st.button("➕ New Chat", use_container_width=True):
        new_sid = create_session()
        st.session_state.active_session_id = new_sid
        active_sid = new_sid
        st.rerun()

    st.divider()
    st.markdown("## 💬 Sessions")

    sorted_sessions = sorted(
        st.session_state.sessions_meta.values(),
        key=lambda s: s["created_at"],
        reverse=True,
    )
    for session in sorted_sessions:
        sid = session["id"]
        is_active = sid == st.session_state.active_session_id
        btn_type = "primary" if is_active else "secondary"
        if st.button(
            session["name"],
            key=f"sess_{sid}",
            use_container_width=True,
            type=btn_type,
        ):
            if not is_active:
                switch_session(sid)
                st.rerun()

    st.divider()
    st.markdown("## 📄 Documents")

    # ── Section 1: File upload ─────────────────────────────────────────────────
    st.markdown("**Upload Files**")
    uploaded_files = st.file_uploader(
        "PDF, TXT, or Markdown",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
        key=f"uploader_{active_sid}",
        label_visibility="collapsed",
    )
    if st.button("Add Files", use_container_width=True, key="btn_add_files"):
        if uploaded_files:
            processed_key = f"processed_files_{active_sid}"
            if processed_key not in st.session_state:
                st.session_state[processed_key] = set()
            with st.spinner("Processing files…"):
                for f in uploaded_files:
                    if f.name in st.session_state[processed_key]:
                        st.info(f"Already loaded: {f.name}")
                        continue
                    suffix = Path(f.name).suffix
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(f.read())
                            tmp_path = tmp.name
                        docs = load_document(tmp_path)
                        for doc in docs:
                            doc.metadata["title"] = Path(f.name).stem
                        add_paper(docs, active_sid)
                        st.session_state[processed_key].add(f.name)
                        st.success(f"Added: {f.name}")
                    except Exception as e:
                        st.error(f"Failed: {f.name} — {e}")
                    finally:
                        if tmp_path:
                            Path(tmp_path).unlink(missing_ok=True)
            st.rerun()
        else:
            st.warning("No files selected.")

    # ── Section 2: Web URL loader ──────────────────────────────────────────────
    st.markdown("**Web Pages**")
    url_input = st.text_area(
        "URLs (one per line)",
        key=f"url_area_{active_sid}",
        height=80,
        label_visibility="collapsed",
        placeholder="https://example.com/paper",
    )
    if st.button("Load URLs", use_container_width=True, key="btn_load_urls"):
        urls = [u.strip() for u in url_input.splitlines() if u.strip()]
        if urls:
            with st.spinner("Loading web pages…"):
                for url in urls:
                    try:
                        docs = load_webpage(url)
                        add_paper(docs, active_sid)
                        st.success(f"Loaded: {url[:60]}")
                    except Exception as e:
                        st.error(f"Failed: {url[:60]} — {e}")
            st.rerun()
        else:
            st.warning("Enter at least one URL.")

    # ── Section 3: ArXiv loader ────────────────────────────────────────────────
    st.markdown("**ArXiv Papers**")
    arxiv_title = st.text_input(
        "Paper title or ArXiv ID",
        key=f"arxiv_input_{active_sid}",
        label_visibility="collapsed",
        placeholder="1706.03762  or  Attention Is All You Need",
    )
    if st.button("Load ArXiv Paper", use_container_width=True, key="btn_load_arxiv"):
        if arxiv_title.strip():
            with st.spinner("Loading from ArXiv…"):
                try:
                    docs = load_arxiv(arxiv_title.strip())
                    add_paper(docs, active_sid)
                    loaded_title = docs[0].metadata.get("title") if docs else arxiv_title.strip()
                    st.success(f"Loaded: {loaded_title}")
                except Exception as e:
                    st.error(f"Failed: {e}")
            st.rerun()
        else:
            st.warning("Enter a paper title or ArXiv ID.")

    # ── Loaded Documents list ──────────────────────────────────────────────────
    st.divider()
    st.markdown("### Loaded Documents")
    try:
        doc_titles = list_papers(active_sid)
    except Exception:
        doc_titles = None
    if doc_titles is None:
        st.caption("Could not load document list — verify Qdrant credentials.")
    elif doc_titles:
        for title in doc_titles:
            st.markdown(f"- {title}")
    else:
        st.caption("No documents loaded yet.")

# ── Main Page Header ───────────────────────────────────────────────────────────
st.title("📚 Papeer — Research Assistant")
st.markdown(
    f"Active Model: `{get_llm_model()}` &nbsp;|&nbsp; "
    f"Embeddings: `{get_embedding_model()}`\n\n"
    "🔍 **Ask questions** from papers &nbsp;·&nbsp; "
    "✅ **Verify claims** against newer work &nbsp;·&nbsp; "
    "🌐 **Live search** with Tavily"
)

# API Key alert if missing
if not os.environ.get("OPENAI_API_KEY"):
    st.warning(
        "⚠️ **OpenAI API Key not set.** Please open **⚙️ Profile & API Settings** in the sidebar "
        "to enter your OpenAI API key and choose your preferred model."
    )

st.divider()

# ── Chat display ───────────────────────────────────────────────────────────────
for msg in st.session_state.chats.get(active_sid, []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            with st.expander(f"📊 Graph state · turn {msg.get('turn', 1)}", expanded=False):
                st.json(msg.get("graph_state", {}))

# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about your papers, verify a claim, or type /btw for side questions…"):
    is_btw = prompt.strip().lower().startswith("/btw")

    if is_btw:
        query = prompt.strip()[4:].strip()

        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption("Side channel — not saved to session history.")

        with st.chat_message("assistant"):
            if not query:
                st.markdown("Please add a question after `/btw`, e.g. `/btw What is attention?`")
            else:
                placeholder = st.empty()
                response_text = ""
                try:
                    for chunk in handle_btw(query):
                        response_text += chunk
                        placeholder.markdown(response_text + "▌")
                    placeholder.markdown(response_text)
                except Exception as e:
                    placeholder.error(f"Error handling /btw request: {e}")
            st.caption("Side channel — not saved to session history.")

    else:
        if active_sid not in st.session_state.chats:
            st.session_state.chats[active_sid] = []
        if active_sid not in st.session_state.turns:
            st.session_state.turns[active_sid] = 0

        is_first_message = len(st.session_state.chats[active_sid]) == 0

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chats[active_sid].append({"role": "user", "content": prompt})
        st.session_state.turns[active_sid] += 1
        current_turn = st.session_state.turns[active_sid]

        if is_first_message:
            maybe_rename_session(active_sid, prompt)

        input_state = {
            "messages": [HumanMessage(content=prompt)],
            "session_id": active_sid,
            "query": prompt,
            "route": None,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "claim_verdict": None,
            "claim_source": None,
            "superseding_papers": [],
            "answer": None,
            "is_relevant": None,
            "rewrite_count": 0,
        }
        config = {"configurable": {"thread_id": active_sid}}

        with st.chat_message("assistant"):
            placeholder = st.empty()
            response_text = ""

            try:
                for chunk, metadata in graph.stream(input_state, config, stream_mode="messages"):
                    if (
                        metadata.get("langgraph_node") == "generate_answer"
                        and hasattr(chunk, "content")
                        and chunk.content
                    ):
                        response_text += chunk.content
                        placeholder.markdown(response_text + "▌")

                if not response_text:
                    final_values = graph.get_state(config).values
                    response_text = final_values.get("answer") or "No response generated."

                placeholder.markdown(response_text)
            except Exception as e:
                response_text = f"⚠️ An error occurred while generating the response: {e}"
                placeholder.error(response_text)

            final_values = graph.get_state(config).values if graph.get_state(config) else {}
            state_snapshot = _serialize_state(final_values)

            with st.expander(f"📊 Graph state · turn {current_turn}", expanded=False):
                st.json(state_snapshot)

        st.session_state.chats[active_sid].append(
            {
                "role": "assistant",
                "content": response_text,
                "graph_state": state_snapshot,
                "turn": current_turn,
            }
        )

        if is_first_message:
            st.rerun()