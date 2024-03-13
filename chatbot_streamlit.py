import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import google.generativeai as genai
from io import StringIO
import ast
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')


df= pd.read_excel('GASTAT scraped links.xlsx')
df['Web Link (PDF)']= df['Web Link (PDF)'].fillna('Not Available')


month_mapping = {
    "Jan": "January",
    "Feb": "February",
    "Mar": "March",
    "Apr": "April",
    "May": "May",
    "Jun": "June",
    "Jul": "July",
    "Aug": "August",
    "Sep": "September",
    "Oct": "October",
    "Nov": "November",
    "Dec": "December"
}

def replace_month_abbreviations_with_full_names(text):
    for abbrev, full in month_mapping.items():
        text = re.sub(r'\b' + abbrev + r'\b', full, text, flags=re.IGNORECASE)
    return text

def create_all_info(row):
    web_name = str(row['Web Name']).lower()
    report_period = str(row['Report Period']).lower()
    report_period = report_period.replace("  ", " ")
    report_period = replace_month_abbreviations_with_full_names(report_period).lower()
    periodicity = str(row['Periodicity'])
    web_name_with_period = f"{web_name} {report_period}" if report_period not in web_name else web_name
    return f"{web_name_with_period} - Periodicty: {periodicity}"

def make_links_clickable(text):
    url_pattern = r'(https?://[^\s/<]+(?:/[^\s<]*)?)'

    seen_urls = set()

    def replace_url(match):
        url = match.group(1)
        following_text = text[match.end():match.end() + 4]
        space_after_url = " " if following_text.startswith("<br>") else ""
        if url not in seen_urls:
            seen_urls.add(url)
            return f'<a href="{url}" target="_blank">{url}</a>{space_after_url}'
        return url

    text = re.sub(url_pattern, replace_url, text)
    return text

def text_preprocess(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove all non-alphabetic and non-numerical characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

#     Initialize the WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize each word
    # Split the text into words for lemmatization
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    # Join the lemmatized words back into a single string
    lemmatized_text = ' '.join(lemmatized_words)

    return text
def model_function(query):
    # reding data and creating all info column
    df['all_info'] = df.apply(create_all_info, axis=1)

    # Query and Data preprocesssing
    df['all_info_preprocessed']= df['all_info'].apply(text_preprocess)
    query= text_preprocess(query)

    # Model Def
    #api configurations
    apikey= 'AIzaSyCEbbTpMCxMid1hCDRjdAu93OsrmTNnckY'
    genai.configure(api_key= apikey)

    # Set up the model
    generation_config = {
      "temperature": 0.5,
      "top_p": 1,
      "top_k": 32,
      "max_output_tokens": 8192,
    }

    safety_settings = [
      {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
      }
    ]

    # Model calling
    model= genai.GenerativeModel(model_name = 'gemini-pro', generation_config = generation_config,
                              safety_settings = safety_settings)

    prompt= f"""You are an expert in converting English questions to python code!
        The dataframe has the name df with the following columns: Web Name, Web Link (XLSX), Web Link (PDF), all_info and all_info_preprocessed
        You will structure the python code based on the keywords available in the asked question, and serach for their match in the web_name column in the matched rows,
        return the web_name, xlsx_link and pdf_link

        for example, Question: do we have data for Riyadh in 2023?
        the keywords will be Riyadh and 2023 and the python code is:\n
import pandas as pd
filtered_data = df[(df['Web Name'].str.contains('Riyadh')) & (df['Web Name'].str.contains('2023'))]
filtered_data = filtered_data[['all_info', 'Web Link (XLSX)', 'Web Link (PDF)']]

        another example, Question: retrieve the data available for prices?
        The keyword should be price and the python code is:
import pandas as pd
filtered_data = df[df['Web Name'].str.contains('price')]
filtered_data['Report Period'] = pd.to_datetime(filtered_data['Report Period'])
filtered_data = filtered_data.sort_values(by=['Web Name', 'Report Period'], ascending=[True, False])
filtered_data = filtered_data.drop_duplicates(subset=['Web Name'], keep='first')
filtered_data = filtered_data[['all_info', 'Web Link (XLSX)', 'Web Link (PDF)']]

        if the question after it asked to retrive all of the years then your answer should be
import pandas as pd
filtered_data = df[df['Web Name'].str.contains('price')]
filtered_data['Report Period'] = pd.to_datetime(filtered_data['Report Period'])
filtered_data = filtered_data.sort_values(by='Report Period', ascending=False)
filtered_data = filtered_data[['all_info', 'Web Link (XLSX)', 'Web Link (PDF)']]

        don't add a print code to your answer
        always import the libraries required
        if the text include greeting then reply with greetings else always start your code with ```python
        if the query contains greetings only then reply with the greeting don't retrieve data for example if the query is 'Hi' then you should reply with something similar to 'Hi, how can I help you today? I'm a conversational bot powered by Gemini, specialized in retrieving data from GASTAT.'

        make sure your code is syntax error free

        if the query contains list all available data then all the data should be retrieved (the names with their link)

        if the data name has more then one row then return the most recent one, but if the query contains listing them all then you should return all of them

        if the query asked contains a month in number or abbreviated then you have to check for the full name of the month

        Be sure to respond in a complete sentence, being comprehensive, including all relevant background information.

        restirct your answer on the data provided only in the df for example if the query is asking about AlUla region, then you shouldn't return anything since this keyword doesn't exist in the data provided.

        The question/query asked is '{query}'
        """
    history=[]
    if len(history) > 0:
        prompt += "\n" + "\n previous conversation: ".join(history)
    chat = model.start_chat()
    response = chat.send_message(prompt)
    history.append(f"Query/Question: {query}\nAnswer: {response.text}")
    try:
        # Try to extract Python code from the text
        code_match = re.search(r'`python(.*?)`', response.text, re.DOTALL)

        if code_match:
            code_snippet = code_match.group(1).strip()
            try:
                ast.parse(code_snippet)
            except IndentationError as ie:
                return (f"Indentation error: {ie}")
            except SyntaxError as se:
                return (f"Syntax error: {se}")
            # Define the local and global context for code execution
            local_context = {'df': df, 'pd': pd}
            global_context = {}
            # Execute the extracted code
            exec(code_snippet, global_context, local_context)
            # Construct the text response with links if 'filtered_data' is in local_context
            if 'filtered_data' in local_context:
                filtered_data = local_context['filtered_data']
                if filtered_data.empty:
                    return 'There is currently no data available on GASTAT for the requested query.'
                else:
                    answer = "Below is the data available on GASTAT for the requested query: \n"

                    for index, row in filtered_data.iterrows():
                        data_name = row['all_info']
                        excel_link = row['Web Link (XLSX)']
                        pdf_link = row['Web Link (PDF)']
                        answer += f"""
Data name: {data_name} \n
Excel link: {excel_link} \n
PDF link: {pdf_link} \n

    """
                    return answer.strip()
            else:
                return "Code executed. No output variable 'filtered_data' found."
        else:
            # If no Python code is found, return the original text
            return response.text
    except Exception as e:
        return f"An error occurred: {e}"


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
        response = model_function(query_input)
        
        st.session_state['qna_history'].append((query_input, response))
    
        st.session_state['clear_input'] = True
    
        st.experimental_rerun()

if __name__ == "__main__":
    main()
