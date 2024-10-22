from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from backend.src.agent.nodes.tool_node import ToolNode
from langgraph.graph import add_messages
from typing import Union

from langchain_core.messages import (
    MessageLikeRepresentation,
)

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][state["current_persona"]][-1].tool_calls
    return {
        "messages": {
            state["current_persona"]: [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes. If it's mot fixable then return the error.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
        }
    }


def create_tool_node_with_fallback(tools: list, message_key:str = None) -> dict:
    return ToolNode(tools=tools, messages_key=message_key).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )
    


Messages = Union[list[MessageLikeRepresentation], MessageLikeRepresentation]


def add_messages_to_dict(left: Messages, right: Messages) -> Messages:
    left_messages = left
    right_messages = right
    dict_key = None
    if isinstance(right,dict):
        for dict_key in right:
            right_messages = right[dict_key]
            left_messages = left.get(dict_key, [])
            if not isinstance(right_messages, list):
                right_messages = [right_messages]
            if not isinstance(left_messages, list):
                left_messages = [left_messages]
            merged = add_messages(left_messages, right_messages)
            left[dict_key] = merged
        return left
    return add_messages(left,right)


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("current_persona")
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        elif isinstance(message, dict):
            message = message[current_state][-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)