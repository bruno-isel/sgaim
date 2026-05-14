from __future__ import annotations

import json
from typing import Any, Dict, Tuple

from pydantic import ValidationError

from schemas import AgentDecision, JsonModeIntent, explain_validation_error, validate_tool_arguments
from tools import execute_tool


TOOLS_SPEC = """
1. get_weather(city: string, date: string)
   Use when the user asks about weather or temperature.

2. calculator(expression: string)
   Use when the user asks to evaluate a mathematical expression.
""".strip()


class MiniAgentController:
    def __init__(self, llm_client: Any) -> None:
        self.llm = llm_client

    def run_json_mode_stage(self, user_message: str) -> Tuple[Dict[str, Any], str]:
        raw = self.llm.json_mode(user_message)
        parsed = json.loads(raw)
        return parsed, raw

    def run_structured_decision_stage(self, user_message: str) -> Tuple[AgentDecision, str]:
        raw = self.llm.structured_decision(user_message, TOOLS_SPEC)
        decision = AgentDecision.model_validate_json(raw)

        if decision.action == "call_tool":
            assert decision.tool_name is not None
            assert decision.arguments is not None
            validate_tool_arguments(decision.tool_name, decision.arguments)

        return decision, raw

    def run_full_loop(self, user_message: str) -> Dict[str, Any]:
        report: Dict[str, Any] = {"user_message": user_message}

        # Stage 1: JSON mode
        json_obj, json_raw = self.run_json_mode_stage(user_message)
        report["json_mode_raw"] = json_raw
        report["json_mode_parsed"] = json_obj

        # Optional validation just for teaching comparison
        try:
            report["json_mode_as_typed_model"] = JsonModeIntent.model_validate(json_obj).model_dump()
        except ValidationError as exc:
            report["json_mode_as_typed_model_error"] = explain_validation_error(exc)

        # Stage 2: structured output / decision
        try:
            decision, decision_raw = self.run_structured_decision_stage(user_message)
            report["decision_raw"] = decision_raw
            report["decision"] = decision.model_dump()
        except ValidationError as exc:
            report["decision_error"] = explain_validation_error(exc)
            return report

        # Stage 3: tool calling loop
        if decision.action == "respond":
            report["final_answer"] = decision.answer
            report["path"] = "direct_response"
            return report

        assert decision.tool_name is not None
        assert decision.arguments is not None

        tool_result = execute_tool(decision.tool_name, decision.arguments)
        report["tool_result"] = tool_result
        report["path"] = f"tool_call:{decision.tool_name}"

        final_answer = self.llm.final_answer(user_message, decision.tool_name, tool_result)
        report["final_answer"] = final_answer
        return report
