"""Bounded model-driven loop. A fresh tool registry and history per question."""
from dataclasses import dataclass
import json
import time
from extension2_finance_agent.core.client import ModelError


@dataclass
class Answer:
    answer: str
    status: str
    tool_calls: int
    model_calls: int
    usage: dict
    elapsed_seconds: float


def answer_question(task, config, client, tools, strategy, emit):
    started = time.perf_counter()
    messages = strategy.messages(task)
    usage = {}
    tool_count = 0
    model_count = 0
    has_evidence = False

    def finish(text, status):
        return Answer(text, status, tool_count, model_count, usage, time.perf_counter() - started)

    for turn in range(config.max_turns):
        allow_tools = tool_count < config.max_tool_calls and turn < config.max_turns - 1
        tick = time.perf_counter()
        model_count += 1
        try:
            response = client.complete(messages, tools.definitions, allow_tools)
        except Exception as exc:
            # Do not log arbitrary provider exceptions, which might contain request secrets.
            event = {"event": "model_error", "turn": turn, "error_type": type(exc).__name__}
            if isinstance(exc, ModelError):
                event["error"] = str(exc)  # Client-generated safe error, never a raw HTTP body.
            emit(event)
            return finish("", "model_error")
        emit({"event": "model_response", "turn": turn, "elapsed_seconds": time.perf_counter() - tick, **response})
        for key, value in response.get("usage", {}).items():
            if type(value) in (int, float):
                usage[key] = usage.get(key, 0) + value
        message = response["message"]
        if response.get("finish_reason") == "length":
            return finish("", "truncated_response")
        calls = message.get("tool_calls") or []
        if not calls:
            content = message.get("content")
            if isinstance(content, str) and content.strip() and has_evidence:
                return finish(content.strip(), "completed")
            if not allow_tools:
                return finish("", "no_evidence" if not has_evidence else "empty_answer")
            messages.extend([message, {"role": "user", "content": "Use the local evidence tools before answering. Return a nonempty final answer grounded in their results."}])
            continue
        if not allow_tools:
            return finish("", "budget_exhausted")
        messages.append(message)
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("id"), str) or not isinstance(call.get("function"), dict):
                return finish("", "invalid_tool_call")
            function = call["function"]
            name = function.get("name", "")
            raw_arguments = function.get("arguments", "")
            tick = time.perf_counter()
            if tool_count >= config.max_tool_calls:
                result = {"ok": False, "data": None, "sources": [], "error": "Tool budget exhausted"}
            else:
                tool_count += 1
                try:
                    arguments = json.loads(raw_arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be a JSON object")
                    result = tools.call(name, arguments)
                except (ValueError, TypeError):
                    result = {"ok": False, "data": None, "sources": [], "error": "Invalid JSON object arguments"}
            has_evidence = has_evidence or (result["ok"] and bool(result.get("sources")))
            emit({"event": "tool_result", "turn": turn, "tool_call_id": call["id"], "name": name,
                  "arguments": raw_arguments, "result": result, "elapsed_seconds": time.perf_counter() - tick})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False, allow_nan=False)})
    return finish("", "budget_exhausted")
