## SMART AI CHATBOT PRO (Jarvis Style)

SMART AI CHATBOT PRO is a **resume-ready Python desktop assistant** with:

- **Modern dark Tkinter UI** with chat bubbles + microphone input
- **Offline AI Mode**: local Ollama chat API with an installed Llama instruct model
- **Online AI Mode**: selected OpenAI or Gemini API, configured independently
- **Memory**: username + favorite personality + chat history (JSON-first)
- **SQLite database**: users + chat history + message_count + favorite_mode + last_login

## Project structure

- `main.py`: CLI + GUI launcher + command routing
- `gui_pro.py`: dark Tkinter app, model controls, chat bubbles, timestamps, typing state
- `gui.py`: compatibility exports for the active GUI
- `chatbot.py`: personality and legacy CLI response helpers
- `voice.py`: text-to-speech + microphone speech-to-text with mute support
- `memory.py`: JSON memory store + in-memory buffer + SQLite integration
- `database.py`: SQLite persistence (`users`, `chat_history`, `settings`, `analytics`)
- `utils.py`: shared utilities (timestamps + slash command parser)
- `config.py`: app configuration and startup directory creation
- `ai_mode.py`: Ollama, OpenAI, and Gemini API routing and response parsing
- `assets/`: UI/static assets directory
- `data/`: runtime JSON/SQLite storage

## Slash commands

- `/help`
- `/time`
- `/date`
- `/joke`
- `/clear`
- `/history`
- `/mode funny`
- `/mode motivator`
- `/mode angry`
- `/mode professional`

## Requirements

- Python **3.10+** recommended (works with modern 3.x)
- Windows: voice features rely on SAPI via `pyttsx3` dependencies

Install:

```bash
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and change `OLLAMA_MODEL` to a model installed on this machine if you choose a different one. `.env` is ignored by Git. The default is `llama3.2:3b`.

## Run

### GUI (recommended)

```bash
python main.py --gui
```

### CLI

```bash
python main.py
```

Backward-compatible:

```bash
python chatbot.py
```

## Modes

### Offline AI Mode

Install Ollama, download a local Llama model, then select **Offline** in the GUI. Requests use `OLLAMA_BASE_URL` and `OLLAMA_MODEL`, send conversation history to `/api/chat`, and do not call a cloud provider.

Windows PowerShell setup:

```powershell
winget install --id Ollama.Ollama -e
if (!(Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
ollama pull llama3.2:3b
python main.py --gui
```

Ollama's desktop app usually starts the local service. If it is not responding, run `ollama serve` in a separate PowerShell window and leave that window open; then run the remaining commands in this one. Do not start a second server if the app is already serving. Edit `OLLAMA_MODEL` in `.env` to match the model shown by `ollama list`.

### Online AI Mode

Select **Online** and choose `openai` or `gemini` in the provider dropdown. The dropdown value is passed directly to the corresponding API client; errors are shown in chat and never fall back to another provider.

Set the key for the provider you selected in `.env` (opened above), then restart the app:

```dotenv
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-4.1-mini
GEMINI_API_KEY=your-gemini-key
GEMINI_MODEL=gemini-2.5-flash
```

OpenAI uses the Responses API and Gemini uses `generateContent`; model names are configurable. Only the selected provider receives a request.

### Verify Each Mode

- **Offline**: select Offline and ask for Python code; confirm the response is generated locally. With Ollama stopped, the app should show an Ollama-specific startup error.
- **Online + OpenAI**: set `OPENAI_API_KEY`, select Online and `openai`, then ask a question.
- **Online + Gemini**: set `GEMINI_API_KEY`, select Online and `gemini`, then ask a question.
- Automated mocked checks: `python -m unittest discover -s tests -v`.

## Data files

- `smartai_memory.json`: username + favorite personality + last conversation history
- `smartai.db` (default): SQLite database for users and chat history

## Notes for GitHub

- Do **not** commit `.venv/` or local DB/history files.
- Add your screenshots and a short demo GIF for a strong portfolio README.

