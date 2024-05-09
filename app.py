import json
import os
import time
import traceback

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

import requests
import streamlit as st

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

safe = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]


# Gemini uses 'model' for assistant; Streamlit uses 'assistant'
def role_to_streamlit(role):
    if role == "model":
        return "assistant"
    else:
        return role


def get_conversation_chain():
    prompt_template = """
        Answer the question clear and precise. If not provided the context return the result as
        "Sorry I dont know the answer", don't provide the wrong answer.
        Context:\n {context}?\n
        Question:\n{question}\n
        Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
    return chain


# Add a Chat history object to Streamlit session state
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

# Display Form Title
st.title("Chat with NPDES")

# Display chat messages from history above current input box
for message in st.session_state.chat.history:

    with st.chat_message(role_to_streamlit(message.role)):
        if message.role == 'user':
            prompt = message.parts[0].text
            st.markdown(prompt)
        else:
            answer = message.parts[0].text
            st.markdown(answer)

        # st.markdown(message.parts[0].text)

# Accept user's next message, add to context, resubmit context to Gemini
if prompt := st.chat_input("What can I help with?"):

    # Display user's last message
    st.chat_message("user").markdown(prompt)

    response = requests.get(f"https://sparcal.sdsc.edu/api/v1/Utility/regulations?search_terms={prompt}")
    datasets = json.loads(response.text)
    st.code(json.dumps(datasets, indent=4))

    docs = [ Document(page_content=dataset["description"]) for dataset in datasets ]
    chain = get_conversation_chain()
    response = chain(
        {"input_documents": docs, "question": prompt},
        return_only_outputs=True
    )
    st.write(response["output_text"], unsafe_allow_html=True)
    
    
