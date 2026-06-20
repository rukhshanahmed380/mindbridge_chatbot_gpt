# MindBridge RAG Chatbot — Group G23
## GPT Version (OpenAI)

---

## Files Required
```
chatbot.py
2_corpus_chunks.csv
requirements.txt
```

---

## Setup — 3 Steps Only

### Step 1 — Install dependency
```bash
pip install openai
```

### Step 2 — Get your OpenAI API Key
- Go to: https://platform.openai.com/api-keys
- Click "Create new secret key"
- Copy the key

### Step 3 — Set key as environment variable
PowerShell:
```powershell
$env:OPENAI_API_KEY = "your_openai_api_key_here"
```

Command Prompt:
```bat
set OPENAI_API_KEY=your_openai_api_key_here
```

---

## Run the Chatbot
```bash
python chatbot.py
```

---

## Commands Inside Chatbot
| Command | Action |
|---|---|
| Type your question | Get answer from corpus |
| `chunks` | See all loaded corpus chunks |
| `quit` or `exit` | Close the chatbot |

---

## Model Options
In `chatbot.py` line 10, you can change the model:
```python
MODEL_ID = "gpt-3.5-turbo"   # cheaper, fast
MODEL_ID = "gpt-4"            # better, costs more
MODEL_ID = "gpt-4o"           # best + fast
```

---

## What Changed from HuggingFace Version
| Old (HuggingFace) | New (OpenAI GPT) |
|---|---|
| `huggingface_hub` library | `openai` library |
| `InferenceClient` | `OpenAI` client |
| `client.text_generation()` | `client.chat.completions.create()` |
| Single string prompt | `messages` list (system + user) |
| Mistral-7B model | gpt-3.5-turbo / gpt-4 |

---

*MindBridge RAG — Group 23 — Campus Support Resources*
*University of Management and Technology — Cloud Computing*
