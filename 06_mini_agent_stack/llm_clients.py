from __future__ import annotations

import json
from typing import Any, Dict, List

import requests


class MockLLMClient:
    """A deterministic mock client for teaching.

    It simulates three stages:
    1. JSON mode
    2. structured decision output
    3. final natural-language answer after tool execution

    This keeps the classroom focus on architecture rather than model variance.
    """

    def json_mode(self, user_message: str) -> str:
        text = user_message.lower()
        if any(word in text for word in ["weather", "temperature", "forecast"]):
            payload = {"intent": "weather_query", "confidence": 0.96}
        elif any(symbol in text for symbol in ["+", "-", "*", "/"]) or "calculate" in text:
            payload = {"intent": "calculation", "confidence": 0.93}
        else:
            payload = {"intent": "general_question", "confidence": 0.62}
        return json.dumps(payload, ensure_ascii=False)

    def structured_decision(self, user_message: str, tools_spec: str = "") -> str:
        text = user_message.lower().strip()

        if "weather" in text or "temperature" in text or "forecast" in text:
            city = "Lisbon" if "lisbon" in text else "Porto" if "porto" in text else "Lisbon"
            date = "tomorrow" if "tomorrow" in text else "today"
            payload = {
                "action": "call_tool",
                "tool_name": "get_weather",
                "arguments": {"city": city, "date": date},
            }
            return json.dumps(payload, ensure_ascii=False)

        if any(symbol in text for symbol in ["+", "-", "*", "/"]) or "calculate" in text:
            expr = text.replace("calculate", "").strip() or "2 + 2"
            payload = {
                "action": "call_tool",
                "tool_name": "calculator",
                "arguments": {"expression": expr},
            }
            return json.dumps(payload, ensure_ascii=False)

        payload = {
            "action": "respond",
            "answer": "This question does not require a tool in the current demo.",
        }
        return json.dumps(payload, ensure_ascii=False)

    def final_answer(self, user_message: str, tool_name: str, tool_result: Dict[str, Any]) -> str:
        if tool_name == "get_weather":
            if tool_result.get("status") == "ok":
                return (
                    f"Weather for {tool_result['city']} on {tool_result['date']}: "
                    f"{tool_result['temperature_c']}°C and {tool_result['condition']}."
                )
            return tool_result.get("message", "No weather information available.")

        if tool_name == "calculator":
            return f"Result: {tool_result['expression']} = {tool_result['result']}"

        return "Tool executed."


class OllamaCompatibleClient:
    """Very small skeleton for an OpenAI-compatible local endpoint.

    This is intentionally minimal. It shows where a real call would fit,
    but the package defaults to the mock client so the example remains
    runnable offline and easy to teach.
    """

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def json_mode(self, user_message: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Respond only with valid JSON with keys 'intent' and 'confidence'. "
                    "Do not include markdown or extra text."
                ),
            },
            {"role": "user", "content": user_message},
        ]
        return self._chat(messages)

    def structured_decision(self, user_message: str, tools_spec: str = "") -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You must respond with valid JSON only.\n"
                    "Schema:\n"
                    "{\n"
                    '  "action": "respond" | "call_tool",\n'
                    '  "answer": string | null,\n'
                    '  "tool_name": "get_weather" | "calculator" | null,\n'
                    '  "arguments": object | null\n'
                    "}\n"
                    "If a tool is needed, set action='call_tool'."
                ),
            },
            {"role": "system", "content": f"Available tools:\n{tools_spec}"},
            {"role": "user", "content": user_message},
        ]
        return self._chat(messages)

    def final_answer(self, user_message: str, tool_name: str, tool_result: Dict[str, Any]) -> str:
        messages = [
            {
                "role": "system",
                "content": "Answer the user naturally and concisely using the tool result.",
            },
            {"role": "user", "content": user_message},
            {"role": "system", "content": f"Tool used: {tool_name}"},
            {"role": "system", "content": f"Tool result: {json.dumps(tool_result, ensure_ascii=False)}"},
        ]
        return self._chat(messages)
