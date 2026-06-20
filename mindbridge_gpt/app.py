from pathlib import Path

from flask import Flask, render_template, request, jsonify
from openai import OpenAI

try:
    from . import chatbot
except ImportError:
    import chatbot

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

corpus = chatbot.load_corpus(str(BASE_DIR / "2_corpus_chunks.csv"))
client = OpenAI(api_key=chatbot.OPENAI_API_KEY or None)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    query = request.form["query"]
    chunks, risk = chatbot.retrieve(query, corpus, top_k=3)
    messages = chatbot.build_messages(query, chunks, risk)
    answer = chatbot.ask_gpt(messages, client)
    return jsonify({"answer": answer, "risk_level": risk})

if __name__ == "__main__":
    app.run(debug=True)
