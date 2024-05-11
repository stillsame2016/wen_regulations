import json
import re

import requests
import streamlit as st
from groq import Groq

Groq_KEY = st.secrets["Groq_KEY"]

def role_to_streamlit(role):
    if role == "model":
        return "assistant"
    else:
        return role


def extract_text_between_question_and_answer(text):
    # Define the pattern to match "Question:" followed by any text until "Answer"
    pattern = r'Question:(.*?)Answer'
    # Use re.DOTALL to make '.' match any character including newlines
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

st.set_page_config(page_title="NPDES")

# Add a Chat history object to Streamlit session state
if "chat" not in st.session_state:
    st.session_state.chat = []

# Display Form Title
# st.title("Chat with NPDES")
st.markdown("### Chat with NPDES ")

# Display chat messages from history above current input box
for message in st.session_state.chat:

    # st.markdown(message)

    with st.chat_message(message['role']):
        st.markdown(message['content'])

        # st.markdown(message.parts[0].text)

# Accept user's next message, add to context, resubmit context to Gemini
if prompt := st.chat_input("What can I help with?"):
    # Display and save the user's input
    st.chat_message("user").markdown(prompt)
    st.session_state.chat.append({"role": "user", "content": prompt})

    response = requests.get(f"https://sparcal.sdsc.edu/api/v1/Utility/regulations?search_terms={prompt}")
    datasets = json.loads(response.text)
    datasets = datasets[0:5]

    context = "\n\n===================\n\n".join([dataset["description"] for dataset in datasets])

    request = f"""
        You are the expert of National Pollution Discharge Elimination System (NPDES). 
        
        The National Pollutant Discharge Elimination System (NPDES) is a regulatory program implemented by the United 
        States Environmental Protection Agency (EPA) to control water pollution. It was established under the Clean 
        Water Act (CWA) to address the discharge of pollutants into the waters of the United States.

        The NPDES program requires permits for any point source that discharges pollutants into navigable waters, 
        which include rivers, lakes, streams, coastal areas, and other bodies of water. Point sources are discrete 
        conveyances such as pipes, ditches, or channels.

        Under the NPDES program, permits are issued to regulate the quantity, quality, and timing of the pollutants 
        discharged into water bodies. These permits include limits on the types and amounts of pollutants that can 
        be discharged, monitoring and reporting requirements, and other conditions to ensure compliance with water 
        quality standards and protect the environment and public health.

        The goal of the NPDES program is to eliminate or minimize the discharge of pollutants into water bodies, 
        thereby improving and maintaining water quality, protecting aquatic ecosystems, and safeguarding human health. 
        It plays a critical role in preventing water pollution and maintaining the integrity of the nation's water 
        resources.
    
        Based on the provided context, using easy understanding language,  answer the question clear and precise with 
        references and explainations. 
        
        Never mention "provided context" or something similar in the answer. 

        If no information is provided in the context,  return the result as "Sorry I dont know 
        the answer", don't provide the wrong answer.

        Context:\n {context}?\n

        Question:\n{prompt}\n
        Answer:
    """

    # st.markdown(request)

    with st.chat_message("assistant"):

        with st.spinner("We are in the process of retrieving the relevant provisions to give you the best possible answer."):
            client = Groq(api_key=Groq_KEY)
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": request}],
                model="llama3-70b-8192")
            result = chat_completion.choices[0].message.content
    
            st.markdown(result)
            st.session_state.chat.append({"role": "assistant", "content": result})
