from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import urllib.error
import urllib.request
from urllib.parse import quote
from dataclasses import dataclass
from typing import Optional, Sequence

from dotenv import load_dotenv


load_dotenv()


class AiMode:
    OFFLINE = "offline"
    ONLINE = "online"


class AiProvider:
    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass
class OnlineAiConfig:
    provider: str = AiProvider.OPENAI
    model: str | None = None
    timeout_s: int = 60


@dataclass
class OllamaConfig:
    base_url: str | None = None
    model: str | None = None
    timeout_s: int = 180


class AiGenerationError(RuntimeError):
    pass


class OnlineAiError(AiGenerationError):
    pass


class OllamaError(AiGenerationError):
    pass


class OllamaUnavailableError(OllamaError):
    pass


class OllamaModelNotFoundError(OllamaError):
    pass


class OllamaModelLoadError(OllamaError):
    pass


class OllamaGenerationError(OllamaError):
    pass


class OllamaTimeoutError(OllamaError):
    pass


class _HttpStatusError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {body[:500]}")


def _http_json(url: str, headers: dict[str, str], payload: dict, timeout_s: int, method: str = "POST") -> dict:
    data = json.dumps(payload).encode("utf-8") if method != "GET" else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            if not isinstance(parsed, dict):
                raise ValueError("The API returned a JSON value instead of an object.")
            return parsed
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        raise _HttpStatusError(e.code, body) from e
    except (TimeoutError, socket.timeout) as e:
        raise TimeoutError("The request timed out.") from e
    except urllib.error.URLError as e:
        if isinstance(e.reason, (TimeoutError, socket.timeout)):
            raise TimeoutError("The request timed out.") from e
        raise ConnectionError(str(e.reason)) from e
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
        raise ValueError(f"The API returned malformed JSON: {e}") from e


def _extract_openai_text(resp: dict) -> str:
    if isinstance(resp, dict) and isinstance(resp.get("output_text"), str) and resp["output_text"].strip():
        return resp["output_text"].strip()

    out = resp.get("output")
    if isinstance(out, list) and out:
        texts: list[str] = []
        for item in out:
            content = item.get("content") if isinstance(item, dict) else None
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") in ("output_text", "text") and isinstance(c.get("text"), str):
                        txt = c["text"].strip()
                        if txt:
                            texts.append(txt)
        if texts:
            return "\n".join(texts)
    raise OnlineAiError("OpenAI response did not contain readable text.")


def _extract_gemini_text(resp: dict) -> str:
    candidates = resp.get("candidates")
    if isinstance(candidates, list) and candidates:
        content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
        parts = content.get("parts") if isinstance(content, dict) else None
        if isinstance(parts, list) and parts:
            texts = [part["text"].strip() for part in parts if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"].strip()]
            if texts:
                return "\n".join(texts)
    raise OnlineAiError("Gemini response did not contain readable text.")


def _online_error(provider: str, error: Exception) -> OnlineAiError:
    if isinstance(error, _HttpStatusError):
        if error.status in (401, 403):
            return OnlineAiError(f"{provider} rejected the API key. Check the configured key and account permissions.")
        if error.status == 429:
            if any(word in error.body.lower() for word in ("quota", "billing", "resource_exhausted")):
                return OnlineAiError(f"{provider} quota is exhausted. Check the account's quota and billing limits.")
            return OnlineAiError(f"{provider} rate limit reached. Wait briefly and try again.")
        return OnlineAiError(f"{provider} request failed with HTTP {error.status}: {error.body[:300]}")
    if isinstance(error, TimeoutError):
        return OnlineAiError(f"{provider} request timed out. Check the connection and try again.")
    if isinstance(error, ConnectionError):
        return OnlineAiError(f"Could not connect to {provider}. Check the internet connection and try again.")
    if isinstance(error, ValueError):
        return OnlineAiError(f"{provider} returned an invalid response: {error}")
    return OnlineAiError(f"{provider} request failed: {error}")


def _ollama_error(error: Exception) -> OllamaError:
    if isinstance(error, TimeoutError):
        return OllamaTimeoutError("Ollama timed out while loading or generating. Try a smaller model or increase available memory.")
    if isinstance(error, (ConnectionError, urllib.error.URLError)):
        executable = shutil.which("ollama") or shutil.which("ollama.exe")
        if executable:
            return OllamaUnavailableError("Ollama is installed but its local service is unavailable. Start the Ollama app or run `ollama serve`, then retry.")
        return OllamaUnavailableError("Ollama is not installed or its local service is unavailable. Install Ollama, start it, then retry.")
    if isinstance(error, ValueError):
        return OllamaGenerationError(f"Ollama returned an invalid response: {error}")
    if isinstance(error, _HttpStatusError):
        body = error.body.lower()
        if "not found" in body and "model" in body:
            return OllamaModelNotFoundError("Ollama does not have the selected model. Download it with `ollama pull <model>` and check OLLAMA_MODEL.")
        if any(word in body for word in ("load model", "loading model", "runner process")):
            return OllamaModelLoadError(f"Ollama could not load the selected model: {error.body[:300]}")
        return OllamaGenerationError(f"Ollama returned HTTP {error.status}: {error.body[:300]}")
    detail = str(error)
    lowered = detail.lower()
    if "not found" in lowered and "model" in lowered:
        return OllamaModelNotFoundError("Ollama does not have the selected model. Download it with `ollama pull <model>` and check OLLAMA_MODEL.")
    if any(word in lowered for word in ("load model", "loading model", "runner process")):
        return OllamaModelLoadError(f"Ollama could not load the selected model: {detail[:300]}")
    return OllamaGenerationError(f"Ollama generation failed: {error}")


def _history_messages(history: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {"role": item["role"], "content": item["content"]}
        for item in history
        if item.get("role") in ("user", "assistant") and item.get("content", "").strip()
    ][-20:]


def _system_prompt(personality: str) -> str:
    styles = {
        "funny": "Use light humor where it fits.",
        "motivator": "Be encouraging without being pushy.",
        "angry": "Use a mildly gruff, dramatic tone while remaining respectful and useful.",
        "professional": "Use a formal, courteous tone.",
        "default": "Use a natural, friendly tone.",
    }
    name = (personality or "default").strip().lower()
    style = styles.get(name, styles["default"])
    return (
        "You are SmartAI Chatbot Pro, a capable general-purpose assistant. "
        "Answer the user's actual request directly, including producing requested code, steps, "
        "explanations, or summaries. Personality affects tone only and must never override or "
        "replace the user's task. Be accurate and say when you are uncertain. "
        f"Selected personality: {name}. {style}"
    )


class OnlineAiClient:
    def __init__(self, config: Optional[OnlineAiConfig] = None) -> None:
        self.config = config or OnlineAiConfig()
        self._log = logging.getLogger(__name__)

    def _model(self, provider: str) -> str:
        configured = self.config.model or os.getenv(f"{provider.upper()}_MODEL", "").strip()
        return configured or ("gpt-4.1-mini" if provider == AiProvider.OPENAI else "gemini-2.5-flash")

    def _openai(self, system_prompt: str, history: Sequence[dict[str, str]], user_text: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise OnlineAiError("OpenAI is selected but OPENAI_API_KEY is missing. Set it in .env or the environment and restart the app.")

        url = "https://api.openai.com/v1/responses"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model(AiProvider.OPENAI),
            "instructions": system_prompt,
            "input": [*_history_messages(history), {"role": "user", "content": user_text}],
        }
        try:
            resp = _http_json(url, headers=headers, payload=payload, timeout_s=self.config.timeout_s)
            return _extract_openai_text(resp)
        except OnlineAiError:
            raise
        except Exception as error:
            raise _online_error("OpenAI", error) from error

    def _gemini(self, system_prompt: str, history: Sequence[dict[str, str]], user_text: str) -> str:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise OnlineAiError("Gemini is selected but GEMINI_API_KEY is missing. Set it in .env or the environment and restart the app.")

        model = quote(self._model(AiProvider.GEMINI), safe="-._")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        contents = [
            {"role": "model" if item["role"] == "assistant" else "user", "parts": [{"text": item["content"]}]}
            for item in _history_messages(history)
        ]
        contents.append({"role": "user", "parts": [{"text": user_text}]})
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
        }
        try:
            resp = _http_json(url, headers=headers, payload=payload, timeout_s=self.config.timeout_s)
            return _extract_gemini_text(resp)
        except OnlineAiError:
            raise
        except Exception as error:
            raise _online_error("Gemini", error) from error

    def generate(self, system_prompt: str, history: Sequence[dict[str, str]], user_text: str) -> str:
        provider = (self.config.provider or "").strip().lower()
        if provider == AiProvider.OPENAI:
            return self._openai(system_prompt, history, user_text)
        if provider == AiProvider.GEMINI:
            return self._gemini(system_prompt, history, user_text)
        raise OnlineAiError(f"Unknown provider '{self.config.provider}'.")


class OllamaAiClient:
    def __init__(self, config: Optional[OllamaConfig] = None) -> None:
        self.config = config or OllamaConfig()

    def _base_url(self) -> str:
        return (self.config.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")

    def _model(self) -> str:
        return self.config.model or os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip() or "llama3.2:3b"

    def generate(self, system_prompt: str, history: Sequence[dict[str, str]], user_text: str) -> str:
        base_url = self._base_url()
        model = self._model()
        try:
            tags = _http_json(f"{base_url}/api/tags", {}, {}, min(self.config.timeout_s, 5), method="GET")
        except Exception as error:
            raise _ollama_error(error) from error

        models = tags.get("models")
        available = {
            str(item.get("name") or item.get("model") or "")
            for item in models
            if isinstance(item, dict)
        } if isinstance(models, list) else set()
        if model not in available:
            raise OllamaModelNotFoundError(
                f"The configured Ollama model '{model}' is not downloaded. Run `ollama pull {model}` or update OLLAMA_MODEL."
            )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(_history_messages(history))
        messages.append({"role": "user", "content": user_text})
        try:
            response = _http_json(
                f"{base_url}/api/chat",
                {"Content-Type": "application/json"},
                {"model": model, "messages": messages, "stream": False},
                self.config.timeout_s,
            )
        except Exception as error:
            raise _ollama_error(error) from error

        if response.get("error"):
            raise _ollama_error(RuntimeError(str(response["error"])))
        message = response.get("message")
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise OllamaGenerationError("Ollama returned no assistant text. Check the selected model and try again.")
        return text.strip()


def generate_reply(
    mode: str,
    provider: str,
    history: Sequence[dict[str, str]],
    user_text: str,
    personality: str,
) -> str:
    system_prompt = _system_prompt(personality)
    if (mode or "").strip().lower() == AiMode.OFFLINE:
        return OllamaAiClient().generate(system_prompt, history, user_text)
    if (mode or "").strip().lower() == AiMode.ONLINE:
        client = OnlineAiClient(OnlineAiConfig(provider=(provider or "").strip().lower()))
        return client.generate(system_prompt, history, user_text)
    raise AiGenerationError(f"Unknown AI mode '{mode}'. Select Offline or Online.")

