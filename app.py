from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as genai
import os

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY        = "AIzaSyCbtijgAii5RBCQ6fmWjucX6BR0zFu_CtE"
MODEL_NAME     = "gemini-3.1-flash-lite-preview"
KNOWLEDGE_FILE = "knowledgebase.txt"

SYSTEM_PROMPT = """You are a helpful and professional travel assistant.
Rules:
- Answer clearly and simply
- Be concise but informative
- If unsure, say you don't know
- Be polite and friendly
"""

# ── Setup Gemini ──────────────────────────────────────────────────────────────
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# ── Load knowledge base once at startup ──────────────────────────────────────
def load_context(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def build_rag_prompt(context, question):
    keywords   = [w.lower() for w in question.split() if len(w) > 3]
    paragraphs = [p.strip() for p in context.split("\n\n") if p.strip()]
    relevant   = [p for p in paragraphs if any(kw in p.lower() for kw in keywords)]
    if not relevant:
        relevant = paragraphs[:3]
    trimmed = "\n\n".join(relevant[:5])
    return f"""Use ONLY the information below to answer the question.
If the answer isn't in the context, say "I don't have that information."

=== KNOWLEDGE BASE ===
{trimmed}
=== END ===

Question: {question}"""

KNOWLEDGE = load_context(KNOWLEDGE_FILE)

# In-memory chat history per session (simple: single global session for now)
chat_history = [
    {"role": "user",  "parts": [SYSTEM_PROMPT]},
    {"role": "model", "parts": ["Understood. I am a travel assistant, ready to help."]},
]

# ── Routes ────────────────────────────────────────────────────────────────────

# Serve the HTML frontend
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

# Chat endpoint — called by the frontend via fetch()
@app.route("/chat", methods=["POST"])
def chat():
    data       = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    # Build prompt: use RAG if knowledge base exists
    if KNOWLEDGE:
        prompt = build_rag_prompt(KNOWLEDGE, user_input)
    else:
        prompt = user_input

    try:
        # Send with full history for memory
        gemini_chat = model.start_chat(history=chat_history.copy())
        response    = gemini_chat.send_message(prompt)
        reply       = response.text

        # Save to history (raw user input, not the RAG-augmented prompt)
        chat_history.append({"role": "user",  "parts": [user_input]})
        chat_history.append({"role": "model", "parts": [reply]})

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Clear chat history
@app.route("/clear", methods=["POST"])
def clear():
    global chat_history
    chat_history = [
        {"role": "user",  "parts": [SYSTEM_PROMPT]},
        {"role": "model", "parts": ["Understood. I am a travel assistant, ready to help."]},
    ]
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    print("Server running at http://localhost:5000")
    app.run(debug=True, port=5000)
