"""Single-node LangGraph baseline and its notebook interface."""

from typing import Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from urbanomy.utils import message_to_text


class SingleAgentState(TypedDict):
    input: str
    output: Any
    log: list[BaseMessage]


def _response_to_output(response: Any) -> Any:
    # LangChain messages are also Pydantic models.
    if isinstance(response, BaseMessage):
        return message_to_text(response.content)
    if isinstance(response, BaseModel):
        return response.model_dump()
    if isinstance(response, dict) and isinstance(response.get("parsed"), BaseModel):
        return response["parsed"].model_dump()
    content = getattr(response, "content", None)
    if content is not None:
        return message_to_text(content)
    return response


def build_single_agent_graph(
    llm: BaseChatModel,
    output_schema: type[BaseModel] | None = None,
) -> CompiledStateGraph:
    """Build a LangGraph baseline with one LLM node."""
    model = llm.with_structured_output(output_schema.model_json_schema()) if output_schema is not None else llm

    def single_agent_node(state: SingleAgentState) -> SingleAgentState:
        messages = [HumanMessage(content=state["input"])]
        response = model.invoke(messages)
        output = _response_to_output(response)
        if output_schema is not None:
            output = output_schema.model_validate(output).model_dump()
        return {
            "input": state["input"],
            "output": output,
            "log": [*state.get("log", []), *messages],
        }

    graph = StateGraph(SingleAgentState)
    graph.add_node("single_agent", single_agent_node)
    graph.add_edge(START, "single_agent")
    graph.add_edge("single_agent", END)
    return graph.compile()


class SingleAgentBaseline:
    """Small wrapper around a one-node LangGraph baseline."""

    def __init__(self, llm: BaseChatModel, output_schema: type[BaseModel] | None = None) -> None:
        self.graph = build_single_agent_graph(llm=llm, output_schema=output_schema)

    def invoke_state(self, text: str, config: dict[str, Any] | None = None) -> SingleAgentState:
        return self.graph.invoke({"input": text, "output": "", "log": []}, config=config)

    def stream(self, text: str, config: dict[str, Any] | None = None):
        return self.graph.stream({"input": text, "output": "", "log": []}, config=config)


__all__ = ["SingleAgentState", "SingleAgentBaseline", "build_single_agent_graph"]
