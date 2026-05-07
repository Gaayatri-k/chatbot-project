import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("Google_Api_key"),
    temperature=0.7
)

# Prompt template
prompt_template = """
You are a helpful AI assistant.

Question: {question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["question"]
)

# Output parser
output_parser = StrOutputParser()

# Chain
def get_qa_chain():

    chain = (
        PROMPT
        | llm
        | output_parser
    )

    return chain


if __name__ == "__main__":

    chain = get_qa_chain()

    response = chain.invoke({
        "question": "What is Python?"
    })

    print(response)