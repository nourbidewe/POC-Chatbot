import streamlit as st
import pandas as pd
import numpy as np
import re
import google.generativeai as genai
# import nltk
# nltk.download('wordnet', quiet=True)
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import wordnet

apikey = "AIzaSyCEbbTpMCxMid1hCDRjdAu93OsrmTNnckY"

def read_excel(path):
    try:
        df = pd.read_excel(path)
        return df
    except FileNotFoundError:
        print("Data file not found. Please ensure the file path is correct.")
    
df= read_excel("Data Sources (new).xlsx")

# df2= read_excel('/Users/user/Desktop/Chatbot/Data Sources (1).xlsx')

try:
    genai.configure(api_key=apikey)
    embed_model = genai.GenerativeModel("models/embedding-001")
    answer_model = genai.GenerativeModel("gemini-pro")
except Exception as e:
    print(f"Error configuring API: {e}")
    
def text_preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # lemmatizer = WordNetLemmatizer()
    # words = text.split()
    # lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    
    # lemmatized_text = ' '.join(lemmatized_words)
    
    # return lemmatized_text
    return text

# def make_links_clickable(text):
#     url_pattern = r'(https?://[^\s/<]+(?:/[^\s<]*)?)'

#     seen_urls = set()

#     def replace_url(match):
#         url = match.group(1)
#         following_text = text[match.end():match.end() + 4]
#         space_after_url = " " if following_text.startswith("<br>") else ""
#         if url not in seen_urls:
#             seen_urls.add(url)
#             return f'<a href="{url}" target="_blank">{url}</a>{space_after_url}'
#         return url

#     text = re.sub(url_pattern, replace_url, text)
#     return text


# def format_response_for_web(text):
#     text = text.replace('\n', '<br>') 
#     bold_pattern = re.compile(r'\*\*(.*?)\*\*')
#     text = bold_pattern.sub(r'<strong>\1</strong>', text)
#     asterisk_items = re.findall(r'\* ([^\*]+)', text)
#     if asterisk_items:
#         for item in set(asterisk_items):  
#             text = text.replace(f"* {item}", f"<li>{item}</li>")
#         text = text.replace("<li>", "<ul><li>", 1)  
#         text = re.sub(r'(<li>[^<]+</li>)', r'\1</ul>', text, 1) 
    
#     return text

def embed_text(text):
    return genai.embed_content(model= 'models/embedding-001',
                                     content= text,
                                    task_type= 'retrieval_document')['embedding']

def query_similarity_score(query, vector):
    query_embedding= embed_text(query)
    return np.dot(query_embedding, vector)

def most_similar_document(query): 
    df['desc_pre']= df['Description'].apply(text_preprocess)
    query= text_preprocess(query)
    df['embeddings']= df['desc_pre'].apply(embed_text)
    df['similarity']= df['embeddings'].apply(lambda vector: query_similarity_score(query, vector))
    df_sorted = df.sort_values("similarity", ascending=False)

    top_similarity = df_sorted.iloc[0]["similarity"]

    similar_documents = df_sorted[df_sorted["similarity"] >= top_similarity - 0.3]
    
    # return list(zip(similar_documents["title"], similar_documents["Description"]))
    return list(zip(similar_documents["Description"]))

def RAG(query):
    history=[]
    documents = most_similar_document(query)  
    if len(history) > 0 and len(documents) == 1:
        # combined_context = "\n".join([f"{title}: {text}" for title, text in documents] + history)
        combined_context = "\n".join([f"{text}" for  text in documents] + history)

    else:
        # Use the original combined_context for first query
        # combined_context = "\n".join(f"{title}: {text}" for title, text in documents)
        combined_context = "\n".join(f"{text}" for text in documents)

        prompt = f"""Answer this query:\n{query}.\nBased on our previous conversation:\n{combined_context}\
            always provide a link to your answer based on the question and answer and don't write the link twice"""
#         prompt = f"""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
#   Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
#   However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
#   strike a friendly and converstional tone. \
#   always provide link or reference to the question asked and don't write the link twice\
#   If the passage is irrelevant to the answer, you may ignore it.
#   QUESTION: '{query}'
#   PASSAGE: '{documents}'

#   ANSWER"""
    model = genai.GenerativeModel("gemini-pro")
    chat= model.start_chat()
    config = genai.types.GenerationConfig(temperature=0.6, max_output_tokens=8192, top_k=10)
    response = chat.send_message(prompt, generation_config=config)
    history.append(f"Query: {query}\nAnswer: {response.text}")
    return response.text
    # return format_response_for_web(response.text)


def main():
    st.title("Data Retrieval Chatbot")
    if 'qna_history' not in st.session_state:
        st.session_state['qna_history'] = []
    
    if st.session_state['qna_history']:
        for q, a in st.session_state['qna_history']:
            st.markdown(f"<span style='color: gray'>User Question:</span><br>", unsafe_allow_html=True)
            st.write(f"{q}")
            st.markdown(f"<span style='color: gray'>Data Retrieval Chatbot:</span><br>", unsafe_allow_html=True)
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
