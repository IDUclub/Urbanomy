from .llms import config


__all__ = ["Agent", "SingleAgentBaseline", "build_single_agent_graph", "config", "init_llm", "llm", "embedding"]


def __getattr__(name: str):
    if name == "Agent":
        from .agent import Agent

        return Agent
    if name in {"init_llm", "llm", "embedding"}:
        from . import llms

        return getattr(llms, name)
    if name in {"SingleAgentBaseline", "build_single_agent_graph"}:
        from . import single_agent

        return getattr(single_agent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
