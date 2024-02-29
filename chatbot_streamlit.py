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
    df_sorted = df.sort_values("similarity", ascending=False)

    # Get top similarity score
    top_similarity = df_sorted.iloc[0]["similarity"]

    # Filter documents meeting the similarity threshold 
    similar_documents = df_sorted[df_sorted["similarity"] >= top_similarity - 0.3]
    
    return list(zip(similar_documents["title"], similar_documents["Descriprion"]))

def RAG(query):
    documents = most_similar_document(query)  # This will return a list of tuples [(title, description), ...]

    if len(documents) == 1:
        # When there's only one document, unpack the tuple
        title, text = documents[0]
        prompt = f"Answer this query:\n{query}.\nOnly use this context to answer:\n{title}: {text}"
    else:
        # For multiple documents, combine their titles and texts for the prompt
        combined_context = "\n".join(f"{title}: {text}" for title, text in documents)
        prompt = f"Answer this query:\n{query}.\nOnly use this context to answer:\n{combined_context}"
    model = genai.GenerativeModel("gemini-pro")
    config = genai.types.GenerationConfig(temperature=0.6, max_output_tokens=8192, top_k=10)
    response = model.generate_content(prompt, generation_config=config)
    return response.text

def main():
    st.title("GASTAT/SADAIA Chatbot")
    if 'qna_history' not in st.session_state:
        st.session_state['qna_history'] = []
    
    # Initialize the input field value in session state if not already set
    if 'query_input' not in st.session_state:
        st.session_state['query_input'] = ''
    
    # Using a form for the input and submit button
    with st.form(key='question_form'):
        # Use the session state for the value of the input field
        query_input = st.text_input("Enter your question:", value=st.session_state['query_input'], key="query_input")
        submit_button = st.form_submit_button('Submit')
    
    if submit_button and query_input:
        # Call the RAG function to get a response for the input question
        response = RAG(query_input)
        
        # Append the question and response to the history
        st.session_state['qna_history'].append((query_input, response))
    
        # Reset the input field in the session state to clear it
        st.session_state['query_input'] = ''
    
        # Indicate that a rerun is needed to clear the form and refresh the page
        st.session_state['need_rerun'] = True
    
    # Display the history of Q&A at the bottom if there are previous Q&As
    if st.session_state['qna_history']:
        for q, a in st.session_state['qna_history']:
            st.write(f"User Question\n: {q}")
            st.write(f"POC Chatbot\n: {a}")
            st.write("-----------------------------")  # Just a separator for readability
    
    # Conditional rerun to avoid recursion
    if 'need_rerun' in st.session_state and st.session_state['need_rerun']:
        # Clear the flag before rerunning to prevent infinite loop
        del st.session_state['need_rerun']
        st.experimental_rerun()

if __name__ == "__main__":
    main()

