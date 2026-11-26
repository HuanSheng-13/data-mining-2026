"""Non-streaming Chat Completions tool calling, using only the stdlib."""
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from extension2_finance_agent.core.env import environment_value as environment_secret


class ModelError(RuntimeError):
    pass


class ChatClient:
    def __init__(self, config):
        self.config = config
        self.key = environment_secret(config.api_key_env)
        if not self.key:
            raise ValueError(f"Missing secret environment variable: {config.api_key_env}")
        url = urllib.parse.urlsplit(config.chat_url)
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("chat_url must not contain credentials, a query or fragment")
        if url.scheme != "https" and not (url.scheme == "http" and url.hostname in ("127.0.0.1", "localhost")):
            raise ValueError("Use HTTPS, or HTTP on localhost")

    def complete(self, messages, tools, allow_tools=True):
        payload = {"model": self.config.model, "messages": messages, "stream": False,
                   "temperature": self.config.temperature, "max_tokens": self.config.max_tokens}
        if allow_tools:
            payload.update(tools=tools, tool_choice="auto")
        if urllib.parse.urlsplit(self.config.chat_url).hostname == "api.deepseek.com":
            payload["thinking"] = {"type": "disabled"}
        request = urllib.request.Request(self.config.chat_url, data=json.dumps(payload).encode("utf-8"),
                                        headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.key})
        for attempt in range(self.config.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    result = json.load(response)
                choice = result["choices"][0]
                message = choice["message"]
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    raise ValueError("Invalid assistant message")
                # Retain only public assistant content and real tool calls, not hidden reasoning.
                public = {"role": "assistant", "content": message.get("content")}
                if message.get("tool_calls"):
                    public["tool_calls"] = message["tool_calls"]
                return {"message": public, "usage": result.get("usage", {}),
                        "finish_reason": choice.get("finish_reason"), "attempts": attempt + 1}
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == self.config.retries:
                    raise ModelError(f"Model HTTP {exc.code}; check service, model, quota and credentials") from None
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.config.retries:
                    raise ModelError("Model connection failed or timed out") from None
            except (KeyError, IndexError, TypeError, ValueError):
                raise ModelError("Model returned an invalid Chat Completions response") from None
            time.sleep(2 ** attempt)
