import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai

apikey = "AIzaSyCEbbTpMCxMid1hCDRjdAu93OsrmTNnckY"
try:
    genai.configure(api_key=apikey)
except Exception as e:
    st.error(f"Error configuring API: {e}")

# Define the models to be used
embed_model = genai.GenerativeModel("models/embedding-001")
answer_model = genai.GenerativeModel("gemini-pro")

# Load data from Excel (replace with your actual file path)
try:
    df = pd.read_excel("Data Sources.xlsx")
except FileNotFoundError:
    st.error("Data file not found. Please ensure the file path is correct.")
# print(df.columns)
    
def embed_text(text):
    # return a list of the embedding tokens 
    return genai.embed_content(model= 'models/embedding-001',
                                     content= text,
                                    task_type= 'retrieval_document')['embedding']

def query_similarity_score(query, vector):
    query_embedding= embed_text(query)
    # return the similarity score using numpy .dot
    return np.dot(query_embedding, vector)

def most_similar_document(query): 
    df['embeddings']= df['Descriprion'].apply(embed_text)
    df['similarity']= df['embeddings'].apply(lambda vector: query_similarity_score(query, vector))
    title= df.sort_values('similarity', ascending= False)[['title', 'Descriprion']].iloc[0]['title']
    text= df.sort_values('similarity', ascending= False)[['title', 'Descriprion']].iloc[0]['Descriprion']
    return title, text

def RAG(query):
    title, text= most_similar_document(query)
    model= genai.GenerativeModel('gemini-pro')
    prompt= f"answer this query:\n {query}. \nOnly use this context to answer:\n {title} :{text}"
    #sourcing
    config= genai.types.GenerationConfig(temperature= 0.6,max_output_tokens= 8192, top_k= 10)
    response= model.generate_content(prompt,generation_config=config)
#     return f'Source Doc Title: {title}\n\n {response.text}'
    return response.text

def main():
    st.title("GASTAT/SADAIA Chatbot")
    if 'qna_history' not in st.session_state:
        st.session_state['qna_history'] = []

    # Display the history of Q&A on top if there are previous Q&As
    if st.session_state['qna_history']:
        for q, a in reversed(st.session_state['qna_history']):
            st.write(f"User Question\n: {q}")
            st.write(f"POC Chatbot\n: {a}")
            st.write("-----------------------------")  # Just a separator for readability

    # Using a form for the input and submit button
    with st.form(key='question_form'):
        query_input = st.text_input("Enter your question:", '', key="query_input")
        submit_button = st.form_submit_button('Submit')

    if submit_button and query_input:
        # Call the RAG function to get a response for the input question
        response = RAG(query_input)
        
        # Append the question and response to the history
        st.session_state['qna_history'].append((query_input, response))

        # Indicate that a rerun is needed to clear the form and refresh the page
        st.session_state['need_rerun'] = True

    # Conditional rerun to avoid recursion
    if 'need_rerun' in st.session_state and st.session_state['need_rerun']:
        # Clear the flag before rerunning to prevent infinite loop
        del st.session_state['need_rerun']
        st.experimental_rerun()

if __name__ == "__main__":
    main()

