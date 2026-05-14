from __future__ import annotations

import json

from controller import MiniAgentController
from llm_clients import MockLLMClient


def pretty(title: str, obj) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    if isinstance(obj, str):
        print(obj)
    else:
        print(json.dumps(obj, indent=2, ensure_ascii=False))


def run_example(user_message: str) -> None:
    controller = MiniAgentController(MockLLMClient())
    report = controller.run_full_loop(user_message)

    pretty("USER MESSAGE", user_message)
    pretty("1) JSON MODE RAW OUTPUT", report.get("json_mode_raw"))
    pretty("2) JSON MODE PARSED", report.get("json_mode_parsed"))

    if "json_mode_as_typed_model" in report:
        pretty("3) OPTIONAL TYPED CHECK OF JSON MODE OUTPUT", report["json_mode_as_typed_model"])
    if "json_mode_as_typed_model_error" in report:
        pretty("3) TYPED CHECK ERROR", report["json_mode_as_typed_model_error"])

    if "decision_error" in report:
        pretty("4) STRUCTURED DECISION ERROR", report["decision_error"])
        return

    pretty("4) STRUCTURED DECISION RAW OUTPUT", report.get("decision_raw"))
    pretty("5) STRUCTURED DECISION PARSED", report.get("decision"))

    if "tool_result" in report:
        pretty("6) TOOL RESULT", report.get("tool_result"))

    pretty("7) FINAL ANSWER", report.get("final_answer"))
    pretty("8) EXECUTION PATH", report.get("path"))


if __name__ == "__main__":
    examples = [
        "What is the weather in Lisbon tomorrow?",
        "Calculate 12 * (3 + 1)",
        "Explain what a transformer is.",
    ]

    for msg in examples:
        run_example(msg)
