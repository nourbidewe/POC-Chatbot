import streamlit as st
import pandas as pd
import numpy as np
import re
# import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import wordnet
import google.generativeai as genai


apikey = "AIzaSyCEbbTpMCxMid1hCDRjdAu93OsrmTNnckY"
try:
    genai.configure(api_key=apikey)
except Exception as e:
    st.error(f"Error configuring API: {e}")

embed_model = genai.GenerativeModel("models/embedding-001")
answer_model = genai.GenerativeModel("gemini-pro")

try:
    df = pd.read_excel("Data Sources.xlsx")
except FileNotFoundError:
    st.error("Data file not found. Please ensure the file path is correct.")
# print(df.columns)
    
def text_preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # lemmatizer = WordNetLemmatizer()
    # words = text.split()
    # lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    
    # lemmatized_text = ' '.join(lemmatized_words)
    
    return text
    
def embed_text(text):
    return genai.embed_content(model= 'models/embedding-001',
                                     content= text,
                                    task_type= 'retrieval_document')['embedding']

def query_similarity_score(query, vector):
    query_embedding= embed_text(query)
    return np.dot(query_embedding, vector)

def most_similar_document(query): 
    df['desc_pre']= df['Descriprion'].apply(text_preprocess)
    query= text_preprocess(query)
    df['embeddings']= df['desc_pre'].apply(embed_text)
    df['similarity']= df['embeddings'].apply(lambda vector: query_similarity_score(query, vector))
    df_sorted = df.sort_values("similarity", ascending=False)

    top_similarity = df_sorted.iloc[0]["similarity"]

    similar_documents = df_sorted[df_sorted["similarity"] >= top_similarity - 0.3]
    
    return list(zip(similar_documents["title"], similar_documents["Descriprion"]))
    
# def RAG(query):

#     documents = most_similar_document(query)  
#     if len(documents) == 1:
#         title, text = documents[0]
#         prompt = f"Answer this query:\n{query}.\nOnly use this context to answer:\n{title}: {text}"
#     else:
#         combined_context = "\n".join(f"{title}: {text}" for title, text in documents)
#         prompt = f"Answer this query:\n{query}.\nOnly use this context to answer:\n{combined_context}"

#     model = genai.GenerativeModel("gemini-pro")
#     config = genai.types.GenerationConfig(temperature=0.6, max_output_tokens=8192, top_k=10)
#     response = model.generate_content(prompt, generation_config=config)
#     return response.text

def RAG(query):
    history=[]
    documents = most_similar_document(query)  
    if len(history) > 0 and len(documents) == 1:
        combined_context = "\n".join([f"{title}: {text}" for title, text in documents] + history)
    else:
        # Use the original combined_context for first query
        combined_context = "\n".join(f"{title}: {text}" for title, text in documents)
        prompt = f"Answer this query:\n{query}.\nBased on our previous conversation:\n{combined_context}"
    model = genai.GenerativeModel("gemini-pro")
    chat= model.start_chat()
    config = genai.types.GenerationConfig(temperature=0.6, max_output_tokens=8192, top_k=10)
    response = chat.send_message(prompt, generation_config=config)
    history.append(f"Query: {query}\nAnswer: {response.text}")
    return response.text

def main():
    st.title("GASTAT/SADAIA Chatbot")
    if 'qna_history' not in st.session_state:
        st.session_state['qna_history'] = []
    
    if st.session_state['qna_history']:
        for q, a in st.session_state['qna_history']:
            st.markdown(f"<span style='color: gray'>User Question:</span><br>", unsafe_allow_html=True)
            st.write(f"{q}")
            st.markdown(f"<span style='color: gray'>POC Chatbot:</span><br>", unsafe_allow_html=True)
            st.write(f"{a}")
            st.write("-----------------------------")  # Just a separator for readability
    
    if 'clear_input' not in st.session_state:
        st.session_state['clear_input'] = False
    
    if st.session_state['clear_input']:
        query_input_value = ''
        st.session_state['clear_input'] = False  # Reset the flag
    else:
        query_input_value = st.session_state.get('query_input', '')
    
    with st.form(key='question_form'):
        query_input = st.text_input("Enter your question:", value=query_input_value, key="query_input")
        submit_button = st.form_submit_button('Submit')
    
    if submit_button and query_input:
        response = RAG(query_input)
        
        st.session_state['qna_history'].append((query_input, response))
    
        st.session_state['clear_input'] = True
    
        st.experimental_rerun()

if __name__ == "__main__":
    main()

