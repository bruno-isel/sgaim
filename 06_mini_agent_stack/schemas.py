from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator


class JsonModeIntent(BaseModel):
    """Simple example used to explain JSON mode.

    In a real JSON-mode setup, the model may produce valid JSON that is only
    parsed with json.loads, without strong schema enforcement. We still define
    the model here to compare the stages side by side.
    """

    intent: str
    confidence: float


class WeatherArgs(BaseModel):
    city: str = Field(..., min_length=1)
    date: str = Field(..., min_length=1)


class CalculatorArgs(BaseModel):
    expression: str = Field(..., min_length=1)


class AgentDecision(BaseModel):
    """Unified schema for the decision phase.

    This is the key teaching move:
    - JSON mode: any valid JSON string is acceptable
    - Structured output: the JSON must satisfy this model
    - Tool calling: the model uses this structure to request an action
    """

    action: Literal["respond", "call_tool"]
    answer: Optional[str] = None
    tool_name: Optional[Literal["get_weather", "calculator"]] = None
    arguments: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def check_consistency(self) -> "AgentDecision":
        if self.action == "respond":
            if not self.answer:
                raise ValueError("When action='respond', the 'answer' field is required.")
            if self.tool_name is not None or self.arguments is not None:
                raise ValueError("Respond actions must not include tool_name or arguments.")

        if self.action == "call_tool":
            if not self.tool_name:
                raise ValueError("When action='call_tool', the 'tool_name' field is required.")
            if self.arguments is None:
                raise ValueError("When action='call_tool', the 'arguments' field is required.")

        return self


TOOL_ARG_SCHEMAS = {
    "get_weather": WeatherArgs,
    "calculator": CalculatorArgs,
}


def validate_tool_arguments(tool_name: str, arguments: Dict[str, Any]) -> BaseModel:
    if tool_name not in TOOL_ARG_SCHEMAS:
        raise ValueError(f"Unknown tool: {tool_name}")
    schema = TOOL_ARG_SCHEMAS[tool_name]
    return schema.model_validate(arguments)


def explain_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", []))
        msg = err.get("msg", "validation error")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return " | ".join(parts)
