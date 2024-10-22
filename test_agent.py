from backend.src.agent.graph import graph
from backend.src.agent.utils import _print_event
import uuid
from langchain_core.messages import ToolMessage

thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}

_printed = set()

while True:
    persona = "General Agent"
    user_query = input("Enter query (type 'quit' to quit): ")
    if user_query=="quit":
        break
    events = graph.stream(
        {"messages": {"General Agent": ("user", user_query)}, "current_persona":persona}, config, stream_mode="values"
    )
    for event in events:
        _print_event(event, _printed)
    snapshot = graph.get_state(config)
    while snapshot.next:
        # We have an interrupt! The agent is trying to use a tool, and the user can approve or deny it
        try:
            user_input = input(
                "Do you approve of the above actions? Type 'y' to continue;"
                " otherwise, explain your requested changed.\n\n"
            )
        except:
            user_input = "y"
        if user_input.strip() == "y":
            result = graph.invoke(
                None,
                config,
                stream_mode="values"
            )
        else:
            result = graph.invoke(
                {
                    "messages": {
                        persona: ToolMessage(
                            tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                            content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                        )
                    }
                },
                config,
            )
        _print_event(result, _printed)
        snapshot = graph.get_state(config)