import json
import requests
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

Groq_KEY = st.secrets["Groq_KEY"]
Groq_KEY_2 = st.secrets["Groq_KEY_2"]

llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=Groq_KEY)
llm2 = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=Groq_KEY_2)

st.set_page_config(page_title="NPDES")

# Add a Chat history object to Streamlit session state
if "chat" not in st.session_state:
    st.session_state.chat = []

# Display Form Title
st.markdown("### Chat with NPDES ")

# Display chat messages from history above current input box
for message in st.session_state.chat:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
st.write("")

user_input = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are the expert of National Pollution 
        Discharge Elimination System (NPDES). 

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

        Based on the provided context, use easy understanding language to answer the question clear and precise with 
        references and explanations. Please don't mention the term "context" in the answer.

        If no information is provided in the context, return the result as "Sorry I dont know 
        the answer", don't provide the wrong answer or a contradictory answer.

        Context:{context}

        Question:{question}?
        
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context"],
)

rag_chain = user_input | llm | StrOutputParser()
rag_chain_2 = user_input | llm2 | StrOutputParser()

# Accept user's next message, add to context, resubmit context to Gemini
if user_input := st.chat_input("What can I help you with?"):
    # Display and save the user's input
    st.chat_message("user").markdown(user_input)
    st.session_state.chat.append({"role": "user", "content": user_input})

    response = requests.get(f"https://sparcal.sdsc.edu/api/v1/Utility/regulations?search_terms={user_input}")
    datasets = json.loads(response.text)
    datasets = datasets[0:5]
    context = "\n".join([dataset["description"] for dataset in datasets])

    with st.chat_message("assistant"):
        with st.spinner(
                "We are in the process of retrieving the relevant provisions to give you the best possible answer."):
            try:
                result = rag_chain.invoke({"question": user_input, "context": context})
            except Exception as e:
                result = rag_chain_2.invoke({"question": user_input, "context": context})

            st.markdown(result)
            st.session_state.chat.append({"role": "assistant", "content": result})

