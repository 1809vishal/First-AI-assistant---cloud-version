"""
STEP 8 (CLOUD VERSION): Deployable Web App
------------------------------------------------
Same assistant as app.py, but built to run on Streamlit Community Cloud
instead of your own laptop, so it's reachable from any browser 24/7,
even when your laptop is off.

Two swaps from the local version:
- Chat model: Ollama (local) -> Groq API (free, hosted, fast)
- Embeddings: nomic-embed-text (Ollama) -> sentence-transformers
  (a small model that runs in-process, no separate server needed)

Setup for LOCAL testing before deploying:
    pip install -r requirements.txt
    Create a file: .streamlit/secrets.toml
    Add this line to it:  GROQ_API_KEY = "your-groq-key-here"
    Run:  streamlit run app.py

For deployment instructions, see the DEPLOY.md file in this folder.

IMPORTANT (free-tier sleep): Streamlit Community Cloud puts free apps to
sleep after 12 hours with no visitors -- this is a platform limit, not
something fixable in code. See DEPLOY.md for a free UptimeRobot
workaround to keep the app awake.
"""

import os
import io
import base64
import json
import hashlib
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from ddgs import DDGS
from pypdf import PdfReader
import docx

CHAT_MODEL = "openai/gpt-oss-120b"  # larger model = more reliable tool calling than the 20b version
VISION_MODEL = "qwen/qwen3.6-27b"  # for understanding photos (Groq's current vision model -- a preview model, may change)
WHISPER_MODEL = "whisper-large-v3-turbo"  # for transcribing spoken questions
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, free, runs in-process
DOCS_FOLDER = "policies"
TOP_K = 2
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
DOCUMENT_EXTENSIONS = {"pdf", "docx", "txt", "csv"}
ALL_UPLOAD_TYPES = sorted(IMAGE_EXTENSIONS | DOCUMENT_EXTENSIONS)

LANGUAGES = [
    "Auto-detect (reply in the same language as the question)",
    "English", "Hindi", "Gujarati", "Spanish", "French", "German",
    "Arabic", "Chinese", "Japanese", "Portuguese", "Russian",
]

st.set_page_config(page_title="My AI Assistant", page_icon="🤖")

# --- Groq client (reads key from Streamlit secrets, not env vars) ---
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])


# --- Build the policy index once per app instance, cached ---
@st.cache_resource
def build_policy_index():
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    chroma_client = chromadb.Client()  # in-memory, rebuilt on each app start
    collection = chroma_client.get_or_create_collection(name="company_policies")

    def chunk_text(text, size, overlap):
        chunks, start = [], 0
        while start < len(text):
            chunks.append(text[start:start + size].strip())
            start += size - overlap
        return [c for c in chunks if c]

    for filename in os.listdir(DOCS_FOLDER):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(DOCS_FOLDER, filename), "r", encoding="utf-8") as f:
            text = f.read()
        for i, chunk in enumerate(chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)):
            embedding = embedder.encode(chunk).tolist()
            collection.upsert(
                ids=[f"{filename}-{i}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": filename}],
            )

    return embedder, collection


embedder, collection = build_policy_index()


# --- Tool implementations ---
def search_web(query: str) -> str:
    try:
        results = DDGS().text(query, max_results=5)
    except Exception as e:
        return f"Search failed: {e}"
    if not results:
        return "No results found."
    return "\n".join(f"- {r['title']}: {r['body']} (source: {r['href']})" for r in results)


def search_company_policies(query: str) -> str:
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=TOP_K)
    if not results["documents"][0]:
        return "No relevant company policy found."
    chunks = []
    for text, metadata in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(f"[Source: {metadata['source']}]\n{text}")
    return "\n\n".join(chunks)


available_functions = {
    "search_web": search_web,
    "search_company_policies": search_company_policies,
}


def analyze_image(image_bytes: bytes, mime_type: str, user_question: str) -> str:
    """Send a photo to Groq's vision model and get back a description or
    an answer to the user's question about it."""
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    prompt_text = user_question.strip() or "Describe what's in this image in detail."
    try:
        response = groq_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}},
                ],
            }],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"(Couldn't analyze the image: {e})"


def transcribe_audio(audio_file) -> str:
    """Send a recorded voice message to Groq's Whisper model and get back
    the transcribed text."""
    try:
        transcription = groq_client.audio.transcriptions.create(
            file=(audio_file.name, audio_file.getvalue()),
            model=WHISPER_MODEL,
        )
        return transcription.text
    except Exception as e:
        return f"(Couldn't transcribe the audio: {e})"


def extract_document_text(file_bytes: bytes, filename: str) -> str:
    """Pull the text content out of an uploaded document (PDF, Word, txt,
    or CSV) so it can be used as context for the assistant."""
    ext = filename.rsplit(".", 1)[-1].lower()
    try:
        if ext == "pdf":
            reader = PdfReader(io.BytesIO(file_bytes))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif ext == "docx":
            document = docx.Document(io.BytesIO(file_bytes))
            text = "\n".join(p.text for p in document.paragraphs)
        elif ext in ("txt", "csv"):
            text = file_bytes.decode("utf-8", errors="ignore")
        else:
            return f"(Unsupported document type: .{ext})"

        text = text.strip()
        if not text:
            return "(No readable text found in this document.)"
        # Cap length so we don't blow past the model's context window
        MAX_CHARS = 6000
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS] + "\n\n[...document truncated...]"
        return text
    except Exception as e:
        return f"(Couldn't read this document: {e})"


# Groq (OpenAI-compatible) needs explicit JSON-schema tool definitions,
# unlike Ollama which could read them straight from Python docstrings.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the live web for current events, recent news, prices, or anything that could have changed since training.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_company_policies",
            "description": "Search internal company policy documents (leave, remote work, expenses, etc).",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The policy question or topic"}},
                "required": ["query"],
            },
        },
    },
]


def build_system_prompt(language: str) -> dict:
    base = (
        "You are a helpful company assistant with two tools available:\n"
        "1. search_web - for current events, live data, or anything that "
        "could have changed since your training.\n"
        "2. search_company_policies - for questions about internal company "
        "rules, HR policy, leave, expenses, or remote work.\n\n"
        "Use the right tool based on what the question is actually about. "
        "If a question needs neither, answer directly. Never guess at "
        "company policy specifics -- always use search_company_policies. "
        "The user may also attach photos or documents -- their extracted "
        "content will appear as [Image content] or [Document: filename] "
        "blocks in the message; treat that as additional context."
    )
    if language and not language.startswith("Auto-detect"):
        base += f"\n\nAlways respond in {language}, regardless of what language the question is in."
    else:
        base += "\n\nRespond in the same language the user's question is written in."
    return {"role": "system", "content": base}


def run_assistant_turn() -> str:
    """Runs the tool-calling flow (web search / policy RAG / direct answer)
    against the current session_state.messages and returns the final
    answer text. Appends any tool-call messages to session_state along
    the way. Shared by both the chat box and the camera capture flow so
    the logic only lives in one place."""
    final_text = None

    try:
        response = groq_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=st.session_state.messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        message = response.choices[0].message
    except Exception:
        try:
            fallback_response = groq_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=st.session_state.messages,
            )
            final_text = fallback_response.choices[0].message.content
        except Exception:
            final_text = (
                "Sorry, I ran into an issue processing that. "
                "Could you try rephrasing your question?"
            )
        message = None

    if message is not None:
        if message.tool_calls:
            assistant_msg = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ],
            }
            st.session_state.messages.append(assistant_msg)

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                try:
                    func_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    func_args = {}
                st.caption(f"🔧 Using tool: {func_name}({func_args})")

                function_to_call = available_functions.get(func_name)
                if not function_to_call:
                    result = f"Unknown function: {func_name}"
                else:
                    query_value = func_args.get("query", "").strip() if isinstance(func_args, dict) else ""
                    if not query_value:
                        result = "No usable search query was provided by the model for this request."
                    else:
                        try:
                            result = function_to_call(query_value)
                        except Exception as e:
                            result = f"Tool execution failed: {e}"

                st.session_state.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": result,
                })

            try:
                final_response = groq_client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=st.session_state.messages,
                )
                final_text = final_response.choices[0].message.content
            except Exception:
                last_tool_result = st.session_state.messages[-1]["content"]
                final_text = (
                    "I ran into trouble putting together a clean answer. "
                    f"Here's what I found:\n\n{last_tool_result}"
                )
        else:
            final_text = message.content

    return final_text


def render_assistant_response():
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            final_text = run_assistant_turn()
            st.markdown(final_text)
            st.session_state.messages.append({"role": "assistant", "content": final_text})


st.title("🤖 My AI Assistant")
st.caption("Live web search + company policy RAG + photos/documents + voice — hosted, available 24/7")

with st.sidebar:
    st.subheader("Settings")
    selected_language = st.selectbox("Response language", LANGUAGES, index=0)

if "messages" not in st.session_state:
    st.session_state.messages = [build_system_prompt(selected_language)]
else:
    st.session_state.messages[0] = build_system_prompt(selected_language)

for msg in st.session_state.messages[1:]:
    if msg["role"] in ("user", "assistant") and msg.get("content"):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# --- Main chat input: text, attach a photo/document, or record voice --
# All three live in the SAME bar via Streamlit's native chat_input --
# tapping "+" on mobile opens the OS's own picker, which offers
# Camera / Photo Library / Files automatically since file_type includes
# image extensions -- no separate menu needed for that.
prompt = st.chat_input(
    "Ask, attach a photo/document, or tap the mic to speak...",
    accept_file=True,
    file_type=ALL_UPLOAD_TYPES,
    accept_audio=True,
)

if prompt:
    user_text = (prompt.text or "").strip()
    extra_context_parts = []

    # --- Handle a recorded voice message ---
    if prompt.audio:
        with st.spinner("Transcribing your voice message..."):
            transcribed = transcribe_audio(prompt.audio)
        user_text = (user_text + " " + transcribed).strip()

    # --- Handle an attached file (image OR document) ---
    image_bytes = None
    attached_filename = None
    if prompt.files:
        uploaded_file = prompt.files[0]
        attached_filename = uploaded_file.name
        ext = attached_filename.rsplit(".", 1)[-1].lower()

        if ext in IMAGE_EXTENSIONS:
            image_bytes = uploaded_file.getvalue()
            image_mime = uploaded_file.type or "image/jpeg"
            with st.spinner("Looking at your photo..."):
                image_description = analyze_image(image_bytes, image_mime, user_text)
            extra_context_parts.append(f"[Image content]: {image_description}")
        else:
            with st.spinner(f"Reading {attached_filename}..."):
                doc_text = extract_document_text(uploaded_file.getvalue(), attached_filename)
            extra_context_parts.append(f"[Document: {attached_filename}]\n{doc_text}")

    combined_text = user_text
    if extra_context_parts:
        combined_text = (combined_text + "\n\n" + "\n\n".join(extra_context_parts)).strip()
    if not combined_text:
        combined_text = "(No question was provided.)"

    st.session_state.messages.append({"role": "user", "content": combined_text})
    with st.chat_message("user"):
        if user_text:
            st.markdown(user_text)
        if prompt.audio:
            st.caption("🎤 Voice message (transcribed above)")
        if image_bytes:
            st.image(image_bytes, width=250)
        elif attached_filename:
            st.caption(f"📄 Attached: {attached_filename}")

    render_assistant_response()
