import streamlit as st
from langchainhelper import create_vectordb, get_qa_chain

st.title("Codebasics FAQ Bot")

# 1. Add the Knowledgebase creation logic
btn = st.button("Create Knowledgebase")
if btn:
    create_vectordb()
    st.success("Knowledgebase created successfully!")

# 2. Use st.cache_resource to prevent reloading the LLM on every question
# This stops the 429 RESOURCE_EXHAUSTED error
@st.cache_resource
def load_chain():
    return get_qa_chain()

question = st.text_input("Question:")

if question:
    # 3. Call the cached version
    chain = load_chain()
    
    # 4. Use .invoke() for 2026 compatibility
    response = chain.invoke({"query": question})
    
    st.header("Answer:")
    st.write(response["result"]) 