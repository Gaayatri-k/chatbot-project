import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import PromptTemplate

# UPDATED: In 2026, RetrievalQA lives in langchain_classic
from langchain_classic.chains import RetrievalQA 

load_dotenv()

# Initialize LLM - Using gemini-3-flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro", 
    google_api_key=os.getenv("Google_Api_key"),
    temperature=0.7
)

embeddings = HuggingFaceEmbeddings()
vectordb_file_path = "faiss_index"

def create_vectordb():
    file_path = "data/codebasics_faqs.csv"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found at: {file_path}")

    loader = CSVLoader(
        file_path=file_path,
        source_column="prompt",
        encoding="utf-8-sig",  
        csv_args={
            'delimiter': ',',
            'quotechar': '"',
        }
    )
    
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    if not os.path.exists(vectordb_file_path):
        print("Index not found. Creating vector database...")
        create_vectordb()
    
    vectordb = FAISS.load_local(
        vectordb_file_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    retriever = vectordb.as_retriever(score_threshold=0.7)
    
    prompt_template = """
    You are a helpful assistant. Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    CONTEXT: {context}

    Question: {question}
    Answer:
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return chain

if __name__ == "__main__":
    if not os.path.exists(vectordb_file_path):
        create_vectordb()
        
    chain = get_qa_chain()
    # In 2026, we use .invoke() instead of the old .__call__()
    response = chain.invoke({"query": "do you provide placement assistance?"})
    print(response["result"])