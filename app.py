import json
import os
import time
import re
import traceback

from groq import Groq
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

import requests
import streamlit as st

# Gemini uses 'model' for assistant; Streamlit uses 'assistant'
def role_to_streamlit(role):
    if role == "model":
        return "assistant"
    else:
        return role


def get_conversation_chain():
    prompt_template = """
        Based on the provided context, Answer the question clear and precise. 
        
        If no information is provided in the context,  return the result as "Sorry I dont know 
        the answer", don't provide the wrong answer.
        
        Context:\n {context}?\n
        
        Question:\n{question}\n
        Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
    return chain


def extract_text_between_question_and_answer(text):
    # Define the pattern to match "Question:" followed by any text until "Answer"
    pattern = r'Question:(.*?)Answer'
    # Use re.DOTALL to make '.' match any character including newlines
    matches = re.findall(pattern, text, re.DOTALL)
    return matches
    

# Add a Chat history object to Streamlit session state
if "chat" not in st.session_state:
    st.session_state.chat = []

# Display Form Title
st.title("Chat with NPDES")

# Display chat messages from history above current input box
for message in st.session_state.chat:

    # st.markdown(message)
    
    with st.chat_message(role_to_streamlit(message.role)):
        if message.role == 'user':
            prompt = message.parts[0].text
            st.markdown(extract_text_between_question_and_answer(prompt)[0])
        else:
            answer = message.parts[0].text
            st.markdown(answer)

        # st.markdown(message.parts[0].text)

# Accept user's next message, add to context, resubmit context to Gemini
if prompt := st.chat_input("What can I help with?"):

    # Display and save the user's input
    st.chat_message("user").markdown(prompt)
    st.session_state.chat.append({"role": "user", "content": prompt})

    response = requests.get(f"https://sparcal.sdsc.edu/api/v1/Utility/regulations?search_terms={prompt}")
    datasets = json.loads(response.text)

    context = "\n\n===================\n\n".join([ dataset["description"] for dataset in datasets ])

    request = f"""
        Based on the provided context, Answer the question clear and precise. 
        
        If no information is provided in the context,  return the result as "Sorry I dont know 
        the answer", don't provide the wrong answer.
        
        Context:\n {context}?\n
        
        Question:\n{prompt}\n
        Answer:
    """

    st.markdown(request)
    
    client = Groq(api_key="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")
    chat_completion = client.chat.completions.create(
        messages=[{ "role": "user",  "content": request }],
        model="llama3-70b-8192")
    result = chat_completion.choices[0].message.content
    result = extract_code_blocks(result)[0] 

    with st.chat_message("assistant"):
        st.markdown(result)
        st.session_state.chat.append({"role": "assistant", "content": result})

    
    
