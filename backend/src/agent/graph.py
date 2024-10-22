from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import ToolMessage, HumanMessage
import json
import os
from openai import OpenAI
from backend.src.agent.state import State
from backend.src.agent.nodes.assistant import Assistant
from backend.src.agent.utils import create_tool_node_with_fallback
from backend.src.agent.tools.general import get_info_from_db, password_reset_tdm, answer_question
from backend.src.agent.rag import populate_vector_db, create_faiss_store


base_url = os.environ.get("MODAL_BASE_URL")
token = os.environ.get("DSBA_LLAMA3_KEY")
api_url = base_url + "/v1"

llm = OpenAI(api_key=token, base_url=api_url)


########### General Agent ###########################

path = os.path.dirname(os.path.abspath(__file__)) + "/"
documents = populate_vector_db(path+"rag_datasets")
create_faiss_store(documents, llm = llm,store_path= path+"faiss_store",rewrite=True)

general_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful Assistant to perform tasks related to general user requests for Ally Bank. "
            "Remember to execute the actions/tools in the order that satisfies the user request. Don't randomize the tool order of execution."
            "\n1) Use the provided tools to find the appropriate tool/tools for the given user request."
        ),
        ("placeholder", "{messages}")
    ]
)

general_safe_tools = [get_info_from_db]

general_sensitive_tools = [password_reset_tdm]

rag_tools = [answer_question]

general_agent = general_prompt | llm.bind_tools(general_safe_tools + general_sensitive_tools + rag_tools)

#######################

####### Graph ################

builder = StateGraph(State)
routes:list = json.load(open(os.path.dirname(os.path.abspath(__file__))+"/../role-values.json"))


def route_to_agent(state:State):
    if state["current_persona"] in routes:
        if state["current_persona"]=="General Agent" or state["salesforce_case"]:
            print("Routing to...",state["current_persona"])
            return state["current_persona"]
        else:
            return END
    raise ValueError(f"{state.current_persona} does not exist")


def route_tools(state: State):
    current_persona = state["current_persona"]
    messages = state["messages"][current_persona]
    next_node = tools_condition(messages)
    # If no tools are invoked, return to the user
    if next_node == END:
        return END
    ai_message = messages[-1]
    first_tool_call = ai_message.tool_calls[0]
    sensitive_tool_names = []
    rag_tool_name = [t.name for t in rag_tools]
    if current_persona == "General Agent":
        if first_tool_call["name"] in rag_tool_name:
            return current_persona + "_rag_tools"   
        sensitive_tool_names = [t.name for t in general_sensitive_tools]
    else:
        raise ValueError(f"{current_persona} tools are not implemented")
    if first_tool_call["name"] in sensitive_tool_names:
        return current_persona + "_sensitive_tools"
    return current_persona + "_safe_tools"

builder.add_conditional_edges(START,route_to_agent,routes)
node_name = routes[0]
builder.add_node(node_name, Assistant(general_agent))
builder.add_node(node_name + "_safe_tools", create_tool_node_with_fallback(general_safe_tools, node_name))
builder.add_node(node_name + "_sensitive_tools",create_tool_node_with_fallback(general_sensitive_tools, node_name))
builder.add_node(node_name + "_rag_tools",create_tool_node_with_fallback(rag_tools, node_name))
builder.add_conditional_edges(
    node_name, route_tools, [node_name + "_safe_tools", node_name + "_sensitive_tools", node_name + "_rag_tools",  END]
)
builder.add_edge(node_name + "_safe_tools", node_name)
builder.add_edge(node_name + "_sensitive_tools", node_name)
builder.add_edge(node_name + "_rag_tools", node_name)

memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=[node_name + "_sensitive_tools" for node_name in routes],
)

graph_image = graph.get_graph().draw_ascii()

with open("graph.txt","w") as f:
    f.write(graph_image)
