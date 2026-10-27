from __future__ import annotations

import logging
import os
import unittest
from unittest.mock import patch

from ai_mode import (
    AiMode,
    AiProvider,
    OnlineAiClient,
    OnlineAiConfig,
    OnlineAiError,
    _HttpStatusError,
    OllamaAiClient,
    OllamaConfig,
    OllamaModelLoadError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaUnavailableError,
    generate_reply,
)


class AiModeTests(unittest.TestCase):
    def test_offline_uses_local_generated_text_and_chat_history(self) -> None:
        code = "items = ['a', 'b']\nitems.append('c')\nitems.insert(1, 'x')\nitems.remove('b')"
        history = [{"role": "user", "content": "Use a list."}, {"role": "assistant", "content": "Sure."}]
        config = OllamaConfig(base_url="http://localhost:11434", model="test-model")
        tags = {"models": [{"name": "test-model"}]}
        generated = {"message": {"role": "assistant", "content": code}}
        with patch("ai_mode._http_json", side_effect=[tags, generated]) as request:
            actual = OllamaAiClient(config).generate("Be helpful.", history, "Demonstrate append, insert, remove")

        self.assertEqual(actual, code)
        self.assertEqual(request.call_args_list[0].kwargs["method"], "GET")
        payload = request.call_args_list[1].args[2]
        self.assertEqual(payload["stream"], False)
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "Be helpful."})
        self.assertEqual(payload["messages"][1:3], history)
        self.assertEqual(payload["messages"][-1]["content"], "Demonstrate append, insert, remove")

    def test_offline_never_constructs_online_client(self) -> None:
        with patch("ai_mode.OllamaAiClient") as local, patch("ai_mode.OnlineAiClient") as online:
            local.return_value.generate.return_value = "generated locally"
            result = generate_reply(AiMode.OFFLINE, AiProvider.GEMINI, [], "hello", "default")

        self.assertEqual(result, "generated locally")
        local.assert_called_once()
        online.assert_not_called()

    def test_online_openai_uses_selected_provider_model_and_history(self) -> None:
        history = [{"role": "user", "content": "Earlier question"}]
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "configured-openai"}), patch(
            "ai_mode._http_json", return_value={"output_text": "OpenAI answer"}
        ) as request:
            result = OnlineAiClient(OnlineAiConfig(provider=AiProvider.OPENAI)).generate(
                "System rules", history, "Current question"
            )

        self.assertEqual(result, "OpenAI answer")
        args = request.call_args.args
        self.assertEqual(args[0], "https://api.openai.com/v1/responses")
        payload = request.call_args.kwargs["payload"]
        self.assertEqual(payload["model"], "configured-openai")
        self.assertEqual(payload["instructions"], "System rules")
        self.assertEqual(payload["input"], [*history, {"role": "user", "content": "Current question"}])

    def test_online_gemini_uses_selected_provider_model_and_history(self) -> None:
        history = [{"role": "assistant", "content": "Earlier answer"}]
        response = {"candidates": [{"content": {"parts": [{"text": "Gemini answer"}]}}]}
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "configured-gemini"}), patch(
            "ai_mode._http_json", return_value=response
        ) as request:
            result = OnlineAiClient(OnlineAiConfig(provider=AiProvider.GEMINI)).generate(
                "System rules", history, "Current question"
            )

        self.assertEqual(result, "Gemini answer")
        args = request.call_args.args
        self.assertIn("/configured-gemini:generateContent", args[0])
        self.assertEqual(request.call_args.kwargs["headers"]["x-goog-api-key"], "test-key")
        self.assertEqual(request.call_args.kwargs["payload"]["contents"], [
            {"role": "model", "parts": [{"text": "Earlier answer"}]},
            {"role": "user", "parts": [{"text": "Current question"}]},
        ])

    def test_switching_modes_changes_the_next_backend(self) -> None:
        with patch("ai_mode.OllamaAiClient") as local, patch("ai_mode.OnlineAiClient") as online:
            local.return_value.generate.return_value = "offline result"
            online.return_value.generate.return_value = "online result"
            offline_result = generate_reply(AiMode.OFFLINE, AiProvider.OPENAI, [], "one", "default")
            online_result = generate_reply(AiMode.ONLINE, AiProvider.OPENAI, [], "two", "default")

        self.assertEqual((offline_result, online_result), ("offline result", "online result"))
        local.assert_called_once()
        online.assert_called_once()

    def test_missing_ollama_service_has_actionable_error(self) -> None:
        config = OllamaConfig(base_url="http://localhost:11434", model="test-model")
        with patch("ai_mode._http_json", side_effect=ConnectionError("connection refused")):
            with self.assertRaises(OllamaUnavailableError) as raised:
                OllamaAiClient(config).generate("system", [], "hello")
        self.assertIn("start", str(raised.exception).lower())

    def test_missing_model_has_download_instruction(self) -> None:
        config = OllamaConfig(base_url="http://localhost:11434", model="not-installed")
        with patch("ai_mode._http_json", return_value={"models": [{"name": "another-model"}]}):
            with self.assertRaises(OllamaModelNotFoundError) as raised:
                OllamaAiClient(config).generate("system", [], "hello")
        self.assertIn("ollama pull not-installed", str(raised.exception))

    def test_model_loading_failure_is_distinct(self) -> None:
        config = OllamaConfig(base_url="http://localhost:11434", model="test-model")
        with patch(
            "ai_mode._http_json",
            side_effect=[{"models": [{"name": "test-model"}]}, _HttpStatusError(500, "failed to load model: insufficient memory")],
        ):
            with self.assertRaises(OllamaModelLoadError):
                OllamaAiClient(config).generate("system", [], "hello")

    def test_ollama_timeout_is_distinct(self) -> None:
        config = OllamaConfig(base_url="http://localhost:11434", model="test-model")
        with patch("ai_mode._http_json", side_effect=TimeoutError("timed out")):
            with self.assertRaises(OllamaTimeoutError):
                OllamaAiClient(config).generate("system", [], "hello")

    def test_missing_provider_key_fails_before_network_request(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), patch("ai_mode._http_json") as request:
            with self.assertRaisesRegex(OnlineAiError, "OPENAI_API_KEY"):
                OnlineAiClient(OnlineAiConfig(provider=AiProvider.OPENAI)).generate("system", [], "hello")
        request.assert_not_called()

    def test_provider_auth_failure_is_not_replaced_by_generic_answer(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "bad-key"}), patch(
            "ai_mode._http_json", side_effect=_HttpStatusError(401, "invalid key")
        ):
            with self.assertRaisesRegex(OnlineAiError, "rejected the API key"):
                OnlineAiClient(OnlineAiConfig(provider=AiProvider.OPENAI)).generate("system", [], "hello")

    def test_model_setting_is_read_from_environment(self) -> None:
        with patch.dict(os.environ, {"OLLAMA_MODEL": "my-local-llama"}):
            client = OllamaAiClient()
            self.assertEqual(client._model(), "my-local-llama")


class DeferredThread:
    started = False

    def __init__(self, *, target, daemon):
        self.target = target
        self.daemon = daemon
        type(self).instance = self

    def start(self) -> None:
        type(self).started = True


class GuiDispatchTests(unittest.TestCase):
    def test_generation_is_started_in_background_without_blocking_send(self) -> None:
        from gui_pro import SmartAIGui

        class FakeValue:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

            def set(self, value):
                self.value = value

        class FakeMemory:
            def __init__(self):
                self.messages = []

            def add(self, role, content, personality):
                self.messages.append(type("Message", (), {"role": role, "content": content})())

        gui = SmartAIGui.__new__(SmartAIGui)
        gui._busy = False
        gui._request_id = 0
        gui._log = logging.getLogger("test")
        gui.input_var = FakeValue("generate something")
        gui.status_var = FakeValue("Ready")
        gui.mode_var = FakeValue(AiMode.OFFLINE)
        gui.provider_var = FakeValue(AiProvider.GEMINI)
        gui.memory = FakeMemory()
        gui.engine = type("Engine", (), {"active_personality": "default"})()
        gui._add_bubble = lambda *args, **kwargs: None
        gui._handle_command = lambda _text: None
        gui._set_inputs_enabled = lambda _enabled: None
        gui._show_typing = lambda: None

        with patch("gui_pro.threading.Thread", DeferredThread), patch("gui_pro.generate_reply") as generate:
            DeferredThread.started = False
            gui._send_message()

        self.assertTrue(DeferredThread.started)
        self.assertTrue(gui._busy)
        generate.assert_not_called()

    def test_clear_chat_discards_an_in_flight_response(self) -> None:
        from gui_pro import SmartAIGui

        class FakeValue:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

            def set(self, value):
                self.value = value

        class FakeMemory:
            def __init__(self):
                self.messages = []

            def add(self, role, content, personality):
                self.messages.append(type("Message", (), {"role": role, "content": content})())

            def start_new_conversation(self):
                self.messages = []

        gui = SmartAIGui.__new__(SmartAIGui)
        gui._busy = False
        gui._request_id = 0
        gui._log = logging.getLogger("test")
        gui.input_var = FakeValue("first request")
        gui.status_var = FakeValue("Ready")
        gui.mode_var = FakeValue(AiMode.OFFLINE)
        gui.provider_var = FakeValue(AiProvider.OPENAI)
        gui.memory = FakeMemory()
        gui.engine = type("Engine", (), {"active_personality": "default"})()
        gui.chat_scroll = type("ChatScroll", (), {"winfo_children": lambda _self: []})()
        gui.input_entry = type("InputEntry", (), {"focus_set": lambda _self: None})()
        gui._add_bubble = lambda role, text, *args, **kwargs: shown.append((role, text))
        gui._handle_command = lambda _text: None
        gui._set_inputs_enabled = lambda _enabled: None
        gui._show_typing = lambda: None
        gui._hide_typing = lambda: None
        gui._startup_welcome = lambda: None
        gui._safe_speak = lambda _text: None
        callbacks = []
        shown = []
        gui.after = lambda _delay, callback: callbacks.append(callback)

        with patch("gui_pro.threading.Thread", DeferredThread), patch(
            "gui_pro.generate_reply", return_value="late answer"
        ):
            gui._send_message()
            gui._clear_chat()
            DeferredThread.instance.target()

        callbacks[0]()
        self.assertEqual(shown, [("user", "first request")])
        self.assertEqual(gui.memory.messages, [])
        self.assertFalse(gui._busy)


if __name__ == "__main__":
    unittest.main()