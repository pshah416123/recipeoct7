# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:21:00 2024

@author: pooja.f.shah
"""

import openai
import langchain
import pypdf
#from langchain.document_loaders import PyPDFLoader
#from langchain.embeddings import OpenAIEmbeddings # for creating embeddings
#from langchain.vectorstores import Chroma # for the vectorization part
from langchain.chains import ChatVectorDBChain # for chatting with the pdf
#from langchain.llms import OpenAI # the LLM model we'll use (CHatGPT)
#from langchain.chat_models import ChatOpenAI
#from langchain.document_loaders import UnstructuredFileLoader
#from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
#from langchain.vectorstores import FAISS 
from langchain.chains.question_answering import load_qa_chain
#from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
import os
import streamlit as st
from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
#from langchain_community.text_splitter import CharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter


#openai.api_key = 'sk-77xclCNx4kWi1H3-BCiruP-R5fGp7sLeqdKTrlSsOgT3BlbkFJQqU3brLhrrG2887xtC66gjvhM4F4KJfVMtNS4Td1AA'

api_key = st.secrets["OPENAI_API_KEY"]


def get_user_input(prompt):
    user_input = st.text_input(prompt)
    if user_input.strip().lower() == 'na':
        return 'any'
    else:
        return f"'{user_input}'"

def generate_meal_recipe():
    protein_preference = get_user_input("What is the minimum number of grams of protein you want your recipe to contain? If you don't have a preference enter 'NA'")
    calorie_preference = get_user_input("About how many calories do you want your meal to contain? If you don't have a preference enter 'NA'")
    ingredient_preference = get_user_input("Do you have any specific ingredients you want to use? If you don't have a preference enter 'NA'")

    prompt = f"Give me a meal and recipe with {protein_preference} grams of protein and about {calorie_preference} calories. Calories can vary by 200. Recipe can use {ingredient_preference} and any other ingredients. List the grams of protein and number of calories this meal contains as well."

    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",  # Use the desired model/engine
        prompt=prompt,
        max_tokens=500,  # Set the desired maximum response length
    )

    generated_text = response.choices[0].text
    st.write(generated_text)


# Function to process PDF and search based on user preferences
def process_pdf_and_search(protein_preference, calorie_preference, ingredient_preference):
    # Construct query using user inputs
    #protein_preference = get_user_input("What is the minimum number of grams of protein you want your recipe to contain? If you don't have a preference enter 'NA' ")
    #calorie_preference = get_user_input("About how many calories do you want your meal to contain? If you don't have a preference enter 'NA' ")
    #ingredient_preference = get_user_input("Do you have any specific ingredients you want to use? If you don't have a preference enter 'NA' ")

    query = f"Give me a meal and recipe with about {protein_preference} grams of protein and about {calorie_preference} calories using {ingredient_preference} and any other ingredients from Pooja's recipes. If you don't know say NA"

    # Read the PDF
    pdf_path = './PoojaRecipes.pdf'
    doc_reader = PdfReader(pdf_path)
    raw_text = ''
    for i, page in enumerate(doc_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    # Splitting up the text into smaller chunks for indexing
    text_splitter = CharacterTextSplitter(        
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # Set up OpenAI embeddings and document search

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    docsearch = FAISS.from_texts(texts, embeddings)

    # Query the document search
    docs = docsearch.similarity_search(query)

    # Load QA chain
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    # Run the QA chain with the documents and query
    result = chain.run(input_documents=docs, question=query)

    st.write(result)


# Streamlit UI
st.title('Recipe Suggestions')

# Radio buttons for recipe choice
recipe_choice = st.radio("Do you want to use one of Pooja's recipes or something from the web?", ('Pooja', 'Web'))

# If 'Pooja' is selected, gather user input preferences
if recipe_choice == 'Pooja':
    protein_preference = st.text_input("What is the minimum number of grams of protein you want your recipe to contain? If you don't have a preference, enter 'NA'.")
    calorie_preference = st.text_input("About how many calories do you want your meal to contain? If you don't have a preference, enter 'NA'.")
    ingredient_preference = st.text_input("Do you have any specific ingredients you want to use? If you don't have a preference, enter 'NA'.")


    # Button to trigger the search
    if st.button("What you got for me Pooj"):
        process_pdf_and_search(protein_preference, calorie_preference, ingredient_preference)

else:
    # If 'Web' is selected, generate a meal recipe
    if st.button("Generate Meal Recipe"):
        generate_meal_recipe()
