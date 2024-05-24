import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from PyPDF2 import PdfReader
import os

from dotenv import load_dotenv

load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'True'

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', '''You are a expert in subjects and solve doubts of everybody based on the content the user will provide you. see that you answer mostly from the given data to you by the user and even provide the text from the data which you referred to provide the answer.'''),
        ('user' , '''Here is the content :{content} analyze and evaluate the content provided to you and answer this question: {question} based on the content provided to you. provide reference paragraph from which you extracted the answer
         
         
         ''')
    ]
)

llm = ChatGoogleGenerativeAI(model='gemini-pro')
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

st.title('Detect PDF e-Book Contents with SkillJo-AI ✨')
upload_files = st.file_uploader('Upload PDF', type='pdf', accept_multiple_files=True)

question = st.text_input('Enter topic from pdf e-book')

if question:
    for cv in upload_files:
        st.write(cv.name)
        extracted_text = ''

        pdf = PdfReader(cv)
        for page_num in range(len(pdf.pages)):
                page = pdf.pages[page_num]
                extracted_text += page.extract_text()
        print(extracted_text)
        st.write(chain.invoke({'question': question, 'content': extracted_text}))
