import streamlit as st
from langchainhelper import get_qa_chain
st.title("Chatbot")

@st.cache_resource
def load_chain():
    return get_qa_chain()

chain = load_chain()

question = st.text_input("Ask anything:")

ask_button = st.button("Ask")

if ask_button and question:

    try:
        with st.spinner("Thinking..."):

            response = chain.invoke({
                "question": question
            })

        st.subheader("Answer")
        st.write(response)

    except Exception as e:

        error_message = str(e)

        if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:

            st.error(
                "API quota exceeded. Try again later or use another API key."
            )

        else:
            st.error(f"Error: {error_message}")