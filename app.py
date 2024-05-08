import json
import os
import time
import traceback

import google.generativeai as genai
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


def justification_markdown(justification_data):
    found_dataset = False
    for dataset in justification_data:
        if dataset['is_relevant']:
            if not found_dataset:
                st.markdown("""
                            Below are the NDP datasets that are semantically closest to your request. 
                            Our searches and justifications are performed using AI. 
                            If you need more relevant datasets, please use other search tools on NDP.
                            """)
                found_dataset = True

            with st.container():
                st.markdown(f"""
                   <hr style="margin: 5px 0px 15px 0px; padding: 0px;" />
                   <table style="border: none; margin-bottom: 5pt;">
                      <tr style="border: none;">
                          <td style="vertical-align: top; border: none; font-weight: bold; margin: 0px; padding: 0px 8px;">
                               Dataset ID:
                          </td>
                          <td style="vertical-align: top; border: none;  margin: 0px 5px; padding: 0px;">
                               {dataset['dataset_id']}
                          </td>
                      </tr>
                      <tr style="border: none;">
                          <td style="vertical-align: top; border: none; font-weight: bold; margin: 0px; padding: 0px 8px;">
                               Title:
                          </td>
                          <td style="vertical-align: top; border: none;  margin: 0px; margin: 0px 5px; padding: 0px;">
                               {dataset['title']}
                           </td>
                      </tr>
                      <tr style="border: none;">
                          <td style="vertical-align: top; border: none; font-weight: bold; margin: 0px; padding: 0px 8px;">
                               Summary:
                           </td>
                          <td style="vertical-align: top; border: none; margin: 0px 5px; padding: 0px;">
                               {dataset['summary']}
                          </td>
                      </tr>
                      <tr style="border: none;">
                          <td style="vertical-align: top; border: none; font-weight: bold; margin: 0px; padding: 0px 8px;">
                               Justification:
                           </td>
                          <td style="vertical-align: top; border: none; margin: 0px 5px; padding: 0px;">
                               {dataset['reason']}
                          </td>
                      </tr>
                   </table>
                """, unsafe_allow_html=True)

            # st.markdown(f"""
            #             **Dataset ID:** {dataset['dataset_id']}
            #             **Title:** {dataset['title']}
            #             **Summary:** {dataset['summary']}
            #             **Justification:** {dataset['reason']}
            #             """)

    if not found_dataset:
        st.markdown(f"""
                    We couldn't locate a dataset closely aligned with your request. 
                    You can try refining your search for further attempts.
                    """)
    return ""


# Add a Chat history object to Streamlit session state
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

# Display Form Title
st.title("Chat with NDP Catalog")

# Display chat messages from history above current input box
for message in st.session_state.chat.history:

    if message.role == 'user':
        # skip the justification request
        prompt = message.parts[0].text.strip()
        if prompt.startswith("The user is looking for datasets with the following keywords"):
            continue

    elif message.role != 'user':
        # skip search terms extraction answer
        answer = message.parts[0].text
        if answer.startswith('```json'):
            json_part = answer.split("\n", 1)[1].rsplit("\n", 1)[0]
            data = json.loads(json_part)
        else:
            data = json.loads(answer)
        if isinstance(data, dict) and "is_search_data" in data.keys() and data["is_search_data"]:
            continue

    with st.chat_message(role_to_streamlit(message.role)):

        if message.role == 'user':
            prompt = message.parts[0].text
            start_index = prompt.find("[--- Start ---]") + len("[--- Start ---]")
            end_index = prompt.find("[--- End ---]")
            prompt = prompt[start_index:end_index].strip()
            st.markdown(prompt)
        else:
            answer = message.parts[0].text
            # print('-' * 70, 'raw answer in history')
            # print(answer)

            if answer.startswith('```json'):
                json_part = answer.split("\n", 1)[1].rsplit("\n", 1)[0]
                data = json.loads(json_part)
            else:
                data = json.loads(answer)

            # print('-' * 70, 'json answer in history')
            # print(data)

            if isinstance(data, dict):
                if not data["is_search_data"]:
                    assistant_response = data["alternative_answer"]
                else:
                    answer = "Searching NDP catalog by the terms:"
                    for term in data['search_terms']:
                        answer = f"{answer}\n - {term}"
                    assistant_response = answer
            else:
                # assistant_response = json.dumps(data, indent=4)
                assistant_response = justification_markdown(data)

            st.markdown(assistant_response)

        # st.markdown(message.parts[0].text)

# Accept user's next message, add to context, resubmit context to Gemini
if prompt := st.chat_input("I'm the NDP Catalog Assistant. Need data or have questions? Just ask!"):

    # Display user's last message
    st.chat_message("user").markdown(prompt)

    query = f"""
      You are an expert of the national data platform catalog for various datasets. 
      You also have general knowledge.
      The following is a question the user is asking:
       
       [--- Start ---]
       {prompt}
       [--- End ---]

       Your main job is to determine if the user is looking for data. 
       If they are looking for data, extract the search terms from the user's request.

       Please answer with a valid JSON string, including the following three fields:
       The boolean field "is_search_data" indicates whether the user is looking for data or not.
       The string list field "search_terms" lists the keywords for which the user is looking for data.
       The string field "alternative_answer" gives your positive answer to the user's input
       if the user is not looking for data.
        
       Please never say "I cannot" or "I could not". 
         
       Please note that the user's request for datasets may appear in the middle of the text, 
       do your best to extract the keywords for which the user is searching for datasets.
         
       Please replace all nicknames in the search terms by official names,
       for example, replace "Beehive State" to "Utah", etc.  
         
       Never deny a user's request to find data. If it is not possible to extract search terms 
       from the user's request, ask the user for further clarification.
       """

    # Send user entry to Gemini and read the response
    asks = 0
    response = None
    while asks < 5:
        try:
            response = st.session_state.chat.send_message(query, safety_settings=safe, )
            break
        except:
            time.sleep(3)
            asks += 1
    if asks == 5:
        st.markdown(
            'This chat has reached the limit on the total number of tokens. Please reload the page to start a new chat.')
        st.stop()

    # Display last
    with st.chat_message("assistant"):

        data = response.text
        # print('-' * 70, 'raw data')
        # print(data)

        if data.startswith('```json'):
            json_part = data.split("\n", 1)[1].rsplit("\n", 1)[0]
            data = json.loads(json_part)
        else:
            data = json.loads(data)

        # print('-' * 70, 'json data')
        # print(data)

        if not data["is_search_data"]:
            assistant_response = data["alternative_answer"]
            st.markdown(assistant_response)
        else:
            answer = "Please wait. We are searching NDP catalog by the terms for you:"
            for term in data['search_terms']:
                answer = f"{answer}\n - {term}"
            assistant_response = answer
            # st.markdown(assistant_response)

            # st.markdown("We found some datasets that might be relevant. Please wait. We are generating the summary")

            search_terms = " ".join(data['search_terms'])
            response = requests.get(f"https://sparcal.sdsc.edu/staging-api/v1/Utility/ndp?search_terms={search_terms}")
            datasets = json.loads(response.text)

            datasets_str = ""
            for dataset in datasets:
                dataset_str = f"""
                 **Dataset Id:** {dataset['dataset_id']}               
                 **Description:** {dataset['description']}
               """

                title, description = dataset['description'].split("|", 1)
                datasets_str += f"""
                  Dataset Id: {dataset['dataset_id']}   
                  Title: {title}            
                  Description: {description} 
               """
                # st.markdown(dataset_str)

            # Send a summary request to Gemini and read the response
            summary_request = f"""
                The user is looking for datasets with the following keywords "{search_terms}"
                     
                The following are the ids and descriptions of some datasets potentially relevant to the user's search terms:
                     {datasets_str}
                
                Provide your answer as a valid JSON list. Each dataset would be one element in this JSON
                list including a string "dataset_id" field for the dataset id,  
                a string field "title" for the title,
                a string field "summary" for summarizing the description with maximum 100 words and 
                without any markdown symbols, 
                a boolean field "is_relevant" for indicating if it is strongly relevant to the search terms and 
                a string field "reason" to explain why these datasets are definitely relevant or irrelevant 
                to the search terms.
                
                Please note that the description may contain the state abbreviation which can be used to exclude 
                datasets. For example, TX usually indicates Texas.
                
                Images and Lidar and DEM data are raster data and not vector data.
                
                If the user requests data for a special region, make sure the region condition is satisfied.
                If the user requests data for a special type, make sure the type condition is satisfied.
                
                If the description contains latitude and longitude, please use them to exclude datasets.
            
                Please note that fire simulation is not earthquake simulation.
            """
            with st.spinner(f"""We're currently conducting a search using the 
                               terms '{" ".join(data['search_terms'])}'.
                               After completing the search, 
                               we'll carefully assess the results to determine 
                               the datasets that best match your semantic criteria. 
                               Your patience is appreciated."""):
                tries = 0
                while tries < 5:
                    try:
                        response = st.session_state.chat.send_message(summary_request, stream=False,
                                                                      safety_settings=safe, )
                        data = response.text

                        # print('='*70)
                        # print(summary_request)
                        # print('-'*70)
                        # print(data)

                        if data.startswith('```json'):
                            json_part = data.split("\n", 1)[1].rsplit("\n", 1)[0]
                            data = json.loads(json_part)
                        else:
                            data = json.loads(data)

                        found = False
                        for item in data:
                            if item['is_relevant']:
                                if not found:
                                    st.markdown("""
                                         Below are the NDP datasets that are semantically closest to your request. 
                                         Our searches and justifications are performed using AI. 
                                         If you need more relevant datasets, please use other search tools on NDP.
                                    """)
                                    found = True

                                with st.container():
                                    st.markdown(f"""
                                        <hr style="margin: 5px 0px 15px 0px; padding: 0px;" />
                                        <table style="border: none; margin-bottom: 5pt;">
                                           <tr style="border: none;">
                                               <td style="vertical-align: top; border: none; font-weight: bold; margin: 0px; padding: 0px 8px;">
                                                    Dataset ID:
                                               </td>
                                               <td style="vertical-align: top; border: none;  margin: 0px 5px; padding: 0px;">
                                                    {item['dataset_id']}
                                               </td>
                                           </tr>
                                           <tr style="border: none;">
                                               <td style="vertical-align: top; border: none; font-weight: bold; margin: 0px; padding: 0px 8px;">
                                                    Title:
                                               </td>
                                               <td style="vertical-align: top; border: none;  margin: 0px; margin: 0px 5px; padding: 0px;">
                                                    {item['title']}
                                                </td>
                                           </tr>
                                           <tr style="border: none;">
                                               <td style="vertical-align: top; border: none; font-weight: bold; margin: 0px; padding: 0px 8px;">
                                                    Summary:
                                                </td>
                                               <td style="vertical-align: top; border: none; margin: 0px 5px; padding: 0px;">
                                                    {item['summary']}
                                               </td>
                                           </tr>
                                           <tr style="border: none;">
                                               <td style="vertical-align: top; border: none; font-weight: bold; margin: 0px; padding: 0px 8px;">
                                                    Justification:
                                                </td>
                                               <td style="vertical-align: top; border: none; margin: 0px 5px; padding: 0px;">
                                                    {item['reason']}
                                               </td>
                                           </tr>
                                        </table>
                                    """, unsafe_allow_html=True)

                                    # st.markdown(f"""
                                    #     **Dataset ID:** {item['dataset_id']}
                                    #     **Title:**  {item['title']}
                                    #     **Summary:**  {item['summary']}
                                    #     **Justification:** {item['reason']}
                                    # """)

                        if not found:
                            st.markdown("""
                                We couldn't locate a dataset closely aligned with your request. 
                                You can try refining your search for further attempts.
                            """)
                        break
                    except Exception as e:
                        print('-' * 70)
                        traceback.print_exc()
                        time.sleep(3)
                        tries += 1
                if tries == 5:
                    st.markdown(
                        'This chat has reached the limit on the total number of tokens. Please reload the page to start a new chat.')
                    st.stop()
