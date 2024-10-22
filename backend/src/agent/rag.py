import os
import faiss
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from uuid import uuid4
from tqdm import tqdm
import shutil
from openai import OpenAI


os.environ['OPENAI_API_KEY'] = 'go-niners'

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
    base_url = os.environ.get("MODAL_BASE_URL")
    token = os.environ.get("DSBA_LLAMA3_KEY")
    api_url = base_url + "/v1"

    llm = OpenAI(api_key=token, base_url=api_url)
    embeddings = OpenAIEmbeddings() 
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(index=index, embedding_function=embeddings,
                        docstore=InMemoryDocstore(), index_to_docstore_id={})
    vectorstore.add_documents(documents, ids=[str(uuid4()) for _ in range(len(documents))])
    vectorstore.save_local(store_path)
    return vectorstore

def load_faiss_store(store_path, llm):
    if os.path.exists(store_path):
        return FAISS.load_local(store_path,allow_dangerous_deserialization=True,embeddings=llm._generate_embeddings)
    raise ValueError(f"{store_path} does not exist for loading FAISS")
        
