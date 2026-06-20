import os
import csv
import re
from openai import OpenAI

# ─────────────────────────────────────────────
#  CONFIG — set OPENAI_API_KEY in environment
# ─────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_ID       = "gpt-3.5-turbo"              # ya "gpt-4" agar access ho
# ─────────────────────────────────────────────


# ── 1. CSV se corpus load karo ───────────────
def load_corpus(csv_path: str) -> list[dict]:
    chunks = []
    with open(csv_path, encoding="utf-8-sig") as f:
        lines = f.readlines()

    header = lines[0].strip().split(",")
    # columns: group_id(0) chunk_id(1) topic(2) category(3) risk_level(4)
    #          title(5) text(6..-4) source_id(-4) allowed_use(-3) blocked_use(-2) language(-1)

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 11:
            continue
        text = ",".join(parts[6:-4]).strip()
        chunks.append({
            "id":         parts[1].strip(),
            "topic":      parts[2].strip(),
            "category":   parts[3].strip(),
            "risk_level": parts[4].strip(),
            "title":      parts[5].strip(),
            "text":       text,
            "source_id":  parts[-4].strip(),
        })
    return chunks


# ── 2. Risk level detect karo ────────────────
CRISIS_WORDS = [
    "assault", "rape", "harass", "suicide", "kill myself",
    "hurt myself", "can't go on", "self-harm", "danger",
    "attacked", "stalking", "emergency", "violence",
    "sexual", "die", "end my life", "not safe",
]
DISTRESS_WORDS = [
    "distress", "discriminat", "threatened", "reporting crime",
    "unsafe", "afraid", "scared",
]
STRESS_WORDS = [
    "stress", "overwhelm", "worried", "anxious", "lonely",
    "isolated", "depressed", "financial help", "can't afford",
    "lost", "confused",
]

def detect_risk(query: str) -> str:
    q = query.lower()
    for w in CRISIS_WORDS:
        if w in q:
            return "L3_CRISIS"
    for w in DISTRESS_WORDS:
        if w in q:
            return "L2_DISTRESS"
    for w in STRESS_WORDS:
        if w in q:
            return "L1_STRESS"
    return "L0_NORMAL"


# ── 3. TF-IDF style retrieval ─────────────────
def tokenize(text: str) -> list[str]:
    return [w for w in re.sub(r"[^a-z0-9\s]", " ", text.lower()).split() if len(w) > 2]

def score_chunk(chunk: dict, query_tokens: list[str]) -> float:
    haystack = f"{chunk['title']} {chunk['text']} {chunk['topic']} {chunk['category']}".lower()
    score = 0.0
    for token in query_tokens:
        occurrences = haystack.count(token)
        if occurrences:
            score += occurrences
        if token in chunk["title"].lower():
            score += 3
        if token in chunk["topic"].lower():
            score += 2
    return score

def retrieve(query: str, corpus: list[dict], top_k: int = 3) -> tuple[list[dict], str]:
    risk = detect_risk(query)
    tokens = tokenize(query)
    scored = []
    for chunk in corpus:
        s = score_chunk(chunk, tokens)
        # crisis queries ko crisis chunks boost milta hai
        if risk in ("L3_CRISIS", "L2_DISTRESS") and chunk["risk_level"] in ("L3_CRISIS", "L2_DISTRESS"):
            s += 10
        scored.append((s, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for s, c in scored[:top_k] if s > 0]
    return top, risk


# ── 4. Messages banao (GPT chat format) ───────
def build_messages(query: str, chunks: list[dict], risk: str) -> list[dict]:
    chunk_text = ""
    for c in chunks:
        chunk_text += f"\n[{c['id']} | {c['category']} | {c['risk_level']}]\n"
        chunk_text += f"Title: {c['title']}\n{c['text']}\n\n"

    safety_note = ""
    if risk == "L3_CRISIS":
        safety_note = (
            "\nIMPORTANT: This is a crisis-level query. "
            "Lead with empathy, list ALL emergency contacts from the chunks, "
            "and end with: 'You are not alone — help is available right now.' "
            "Always mention 911 as a fallback."
        )
    elif risk == "L2_DISTRESS":
        safety_note = "\nThis student shows signs of distress. Be supportive and provide concrete resource referrals."
    elif risk == "L1_STRESS":
        safety_note = "\nThe student seems stressed. Be warm, practical, and connect them to the right resource."

    system_prompt = (
        "You are MindBridge RAG, a student support chatbot for Group G23.\n"
        "Answer ONLY using the corpus chunks provided in the user message. "
        "Be accurate, specific, and grounded.\n"
        "At the end of your answer, always add a 'Sources:' line listing the chunk IDs you used."
        + safety_note
    )

    user_message = (
        f"CORPUS CHUNKS:\n{chunk_text}\n"
        f"STUDENT QUESTION: {query}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]


# ── 5. OpenAI GPT API call ────────────────────
def ask_gpt(messages: list[dict], client: OpenAI) -> str:
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_tokens=600,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# ── 6. Display helpers ────────────────────────
RISK_COLORS = {
    "L0_NORMAL":   "\033[92m",   # green
    "L1_STRESS":   "\033[93m",   # yellow
    "L2_DISTRESS": "\033[91m",   # red
    "L3_CRISIS":   "\033[95m",   # magenta
}
RESET = "\033[0m"
BOLD  = "\033[1m"
CYAN  = "\033[96m"
GRAY  = "\033[90m"

def print_banner():
    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════╗
║        MindBridge RAG Chatbot — Group G23            ║
║   50 Chunks · 5 Categories · S2 Safety-Aware RAG     ║
║   Model: {MODEL_ID} via OpenAI              ║
╚══════════════════════════════════════════════════════╝{RESET}
Type your question and press Enter.
Type {BOLD}'quit'{RESET} or {BOLD}'exit'{RESET} to close.
Type {BOLD}'chunks'{RESET} to see how many corpus chunks are loaded.
""")

def print_retrieved(chunks: list[dict], risk: str):
    color = RISK_COLORS.get(risk, "")
    print(f"\n{GRAY}┌─ RAG Retrieved {len(chunks)} chunk(s) │ Risk: {color}{risk}{RESET}{GRAY} ─────────────────{RESET}")
    for c in chunks:
        print(f"{GRAY}│  {c['id']} — {c['title'][:55]}{RESET}")
    print(f"{GRAY}└───────────────────────────────────────────────────────{RESET}")

def print_answer(text: str):
    print(f"\n{BOLD}MindBridge:{RESET} {text}\n")
    print("─" * 55)


# ── 7. Main chat loop ─────────────────────────
def main():
    # CSV file path
    csv_path = os.path.join(os.path.dirname(__file__), "2_corpus_chunks.csv")
    if not os.path.exists(csv_path):
        print(f"\n❌  Error: '{csv_path}' not found.\n"
              "    Make sure 2_corpus_chunks.csv is in the same folder as chatbot.py\n")
        return

    # Corpus load karo
    corpus = load_corpus(csv_path)
    if not corpus:
        print("❌  No chunks loaded. Check your CSV file.")
        return

    # OpenAI API key check
    if not OPENAI_API_KEY:
        print("\n⚠️   OPENAI_API_KEY not set in environment!")
        print("    Set it before running the app. Example (PowerShell):\n")
        print('    $env:OPENAI_API_KEY = "your_openai_api_key_here"\n')
        print("    OpenAI key yahan se milegi: https://platform.openai.com/api-keys\n")
        return

    # OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)

    print_banner()
    print(f"✅  {len(corpus)} chunks loaded from corpus.\n")

    while True:
        try:
            user_input = input(f"{BOLD}You:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "chunks":
            print(f"\n📦  {len(corpus)} chunks loaded:\n")
            cats = {}
            for c in corpus:
                cats.setdefault(c["category"], []).append(c["id"])
            for cat, ids in cats.items():
                print(f"  {cat}: {len(ids)} chunks ({', '.join(ids[:3])}{'...' if len(ids)>3 else ''})")
            print()
            continue

        # RAG pipeline
        chunks, risk = retrieve(user_input, corpus, top_k=3)
        print_retrieved(chunks, risk)

        if not chunks:
            print_answer("I could not find relevant information in the corpus for your question. Please try rephrasing.")
            continue

        print(f"{GRAY}  Generating answer...{RESET}", end="\r")
        messages = build_messages(user_input, chunks, risk)

        try:
            answer = ask_gpt(messages, client)
            print_answer(answer)
        except Exception as e:
            print_answer(f"❌ API Error: {e}\n\nCheck your OPENAI_API_KEY and internet connection.")


if __name__ == "__main__":
    main()
