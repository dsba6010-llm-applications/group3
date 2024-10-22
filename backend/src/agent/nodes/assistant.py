from langchain_core.runnables import Runnable, RunnableConfig
from backend.src.agent.state import State

class Assistant:
    def __init__(self, runnable: Runnable, persona: str=None):
        self.runnable = runnable
        self.persona = persona

    def __call__(self, state: State, config: RunnableConfig):
        retried = 0
        while True:
            current_persona = self.persona or state["current_persona"]
            _state = {**state, "messages": state["messages"][current_persona]}
            result = self.runnable.invoke(_state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                retried += 1
                if retried>=3:
                    print("Retried 3 times for empty response")
                    break
                
                state["messages"][current_persona] = state["messages"][current_persona] + [("user", "Respond with a real output.")]
            else:
                break
        return {"messages": {current_persona: result}}