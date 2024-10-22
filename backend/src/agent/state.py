from typing import TypedDict, Annotated, Dict, Any, List
# from langgraph.graph import add_messages
from backend.src.agent.utils import add_messages_to_dict
from langchain_core.messages import AnyMessage
class State(TypedDict):
    messages: Annotated[Dict[str,list[AnyMessage]], add_messages_to_dict]
    salesforce_case: Dict[str,Any]
    salesforce_cases: List[Dict[str,Any]]
    current_persona: str
 