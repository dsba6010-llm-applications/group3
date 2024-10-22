import os
import faiss
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from uuid import uuid4
from tqdm import tqdm
import shutil

def populate_vector_db(directory):
    print("Populating Vector DB...")
    documents = []
    splitter = RecursiveCharacterTextSplitter(separators=["##"], chunk_size=1000, chunk_overlap=200)
    chunk_id = 0
    filenames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f))]
    for filename in tqdm(filenames, desc="Processing file: "):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
            content = f.read()
            chunks = splitter.split_text(content)
            file = "".join(filename.split(".")[:-1])
            output_path = os.path.join(directory, file)
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            for chunk in chunks:
                os.makedirs(output_path, exist_ok=True)
                open(output_path+f"/{chunk_id}.txt","w").write(chunk)
                documents.append(Document(page_content=chunk, metadata={"filename": filename, "chunk_id":chunk_id}))
                chunk_id += 1
    return documents

def create_faiss_store(documents, llm, store_path="faiss_index", embedding_size=1536, rewrite=False):
    if os.path.exists(store_path) and (not rewrite):
        return FAISS.load_local(store_path,allow_dangerous_deserialization=True,embeddings=llm._generate_embeddings)
    print("Creating FAISS store...")
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(index=index, embedding_function=llm._generate_embeddings,
                        docstore=InMemoryDocstore(), index_to_docstore_id={})
    vectorstore.add_documents(documents, ids=[str(uuid4()) for _ in range(len(documents))])
    vectorstore.save_local(store_path)
    return vectorstore

def load_faiss_store(store_path, llm):
    if os.path.exists(store_path):
        return FAISS.load_local(store_path,allow_dangerous_deserialization=True,embeddings=llm._generate_embeddings)
    raise ValueError(f"{store_path} does not exist for loading FAISS")
        

'''
def generate_response(vectorstore, query):
    relevant_docs = retrieve_with_relevance_ranking(vectorstore, query)
    relevant_docs.sort(key=lambda x:x[1],reverse=True)
    context = "\n\n".join([doc[0].page_content for doc in relevant_docs])
    if len(relevant_docs)>0:
        full_query = f"Question: {query}\nInstructions: Ignore any Markdown syntax such as #, | or special characters in the context. Use only the relevant information in the context to answer the question concisely and clearly. Do not repeat context verbatim, and provide meaningful response based on the context.\nContext:\n{context}"
    else:
        full_query = query
    #result = ally_chat.invoke({
    #    "messages": [
    #        SystemMessage(content="You are a helpful AI assistant."),
    #       HumanMessage(content=full_query)
    #   ]
    #})
        response = ally_chat._generate([
                SystemMessage(content="You are a helpful AI assistant."),
                HumanMessage(content=full_query)
            ])
        result = response.generations[0].message.content
    
    with open("rag_log.txt", "+a") as f:
        f.write(f"\nQuery: {query}")
        f.write(f"\nRetrieved {len(relevant_docs)} docs")
        f.write(f"\n\nRelevant Docs :: \n{relevant_docs}\n")
        f.write(f"\n\nResponse: \n{result}\n")
        f.write("="*50)
    return result

print("[INFO] Creating chunks...")
documents = create_chunks_from_markdown('./md_files/')
print(f"[INFO] Total chunks::{len(documents)}")
print("[INFO] Storing in FAISS....")
faiss_vectorstore = create_faiss_index(documents)
print("[INFO] Testing...\n")
while True:
    user_query = str(input("Enter Query (quit() to exit): "))
    if user_query=="quit()":
        break
    if user_query:
        response = generate_response(faiss_vectorstore, user_query)
        print("[INFO] Model response:", response["messages"][-1])
'''