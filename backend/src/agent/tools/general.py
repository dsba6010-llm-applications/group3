import requests
from langchain_core.tools import tool
import json
from typing import Any, Dict
from langchain_core.runnables import RunnableConfig
from agent.ally_llm import AllyChat
import os

from operator import itemgetter
from langchain.schema.runnable import RunnableMap    
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from agent.rag import load_faiss_store

def getTDMAccessToken():
    url = "https://qa.api.ally.com/v1/access/token"
    payload = 'grant_type=client_credentials'
    headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Authorization': 'Basic NFNhbzZseXpPUXhLMFBScExSZUdjRmQxemE0VldmYWg6TG8zczFNR1BkckJ1dlVSZQ=='
                }
 
    response = requests.request("POST", url, headers=headers, data=payload)
    if response.status_code==200 and 'access_token' in response.json():
        print("got access token")
        return response.json()['access_token']
    else:
        return 'Error'

@tool
def password_reset_tdm(env:str,
                         username:str, 
                         newPassword:str) -> str:
    """Changes the password of the given user name on the given ENV using TDM API.

    Args:
        env (str): The environment where the user password needs to change. It could be QA, DEV, PROD etc.
        username (str): User name of the user
        newPassword (str): New password to change

    Returns:
        str: Status of the password change
    """
    Auth = getTDMAccessToken()
    print(Auth)
    if 'Error' not in Auth:          
        url = 'https://qa.api.ally.com/tdm-api/passwordChange'
        payload ={"testingEnv":env.upper(),"userName":username,"newPassword":newPassword}
        headers = {
        'Authorization': 'Bearer '+ Auth
        }
        response = requests.request("POST", url, json=payload,headers=headers)
        if response.status_code == 200:
            return response.json()['msg']
        else:
            print(response.status_code)
            print(response.text)
            return 'Error occured at the backend during password change.'
 
    else:
        return "Error occured at the backend during password change."


@tool
def get_info_from_db(name:str, config: RunnableConfig) -> Dict[str, Any]:
    """Fetches and returns the user information from DB
    
    Args:
        name (str): full name of the user

    Returns:
        Dict[str, Any]: A dictionary with all information about a person
    """
    db_path = os.path.dirname(os.path.abspath(__file__)) + "/../db/profile.json"
    
    print("Getting user info from : ",db_path)
    if not name:
        print("No name was provided, getting from config..")
        configuration = config.get("configurable", {})
        name = configuration.get("name", None)
    print("Name :: ",name)
        

    with open(
        db_path,
        "r",
    ) as file:
        users_info = json.load(file)

        for user_info in users_info:
            if f"{user_info['first_name']} {user_info['last_name']}" == name:
                print("found user :: ",user_info)
                return user_info
        return f"Couldn't find user {name} in DB"


@tool
def answer_question(query: str):
    """Fetches and returns information using RAG for user query for general purposes and searches.
    
    Args:
        user_query (str): Query to search for answers

    Returns:
        str: response of user query fetched through RAG
    """
    llm = AllyChat()
    path = os.path.dirname(os.path.abspath(__file__)) + "/../"
    db = load_faiss_store(path+"faiss_store/", llm)
    query_embedding = llm._generate_embeddings(query)
    docs = db.max_marginal_relevance_search_with_score_by_vector(
        embedding=query_embedding, k=5, lambda_mult=0.1, fetch_k=30 
    )
    docs.sort(key=lambda x:x[1],reverse=True)
    context = "\n\n".join([doc[0].page_content for doc in docs])
    template = [
        ("system", 
        "Use the following context to answer the question at the end."
        " Process the context by removing any special characters that might be from a Markdown or other files."
        " If you don't know the answer, just say that you don't know, don't try to make up an answer."
        "\nContext:\n"
        "{context}"
    ),
    ("user",
     "Question: {question}"
     "\nHelpful Answer:")
    ]
    rag_prompt_custom = ChatPromptTemplate.from_messages(template)
    chain = (
        rag_prompt_custom
        | llm
    )
    
    response = chain.invoke({
            "context":context,
            "question": query
        })
    
    return response.content
