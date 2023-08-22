import os
import openai
import sys
from langchain.document_loaders import PyPDFLoader
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS, Chroma
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.llms import OpenAI, HuggingFaceHub
import requests
import gradio as gr
from langchain.document_loaders import PyMuPDFLoader, PyPDFLoader
# Get your API keys from openai, you will need to create an account.
# Here is the link to get the keys: https://platform.openai.com/account/billing/overview
import fitz
from PIL import Image
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, DocArrayInMemorySearch
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, load_index_from_storage
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import panel as pn
import param
from dotenv import load_dotenv
load_dotenv()

if os.path.exists(os.environ["INDEX_FILE"]):
    print("Found existing index.")
    # index = GPTVectorStoreIndex.load_from_disk(os.environ["INDEX_FILE"])
    embeddings=OpenAIEmbeddings()
    db = FAISS.load_local(os.environ["INDEX_FILE"], embeddings)
else:
    loader=PyPDFLoader(file)
    pages= loader.load()
    text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separator='\n'
    )
    Str_pages=''
    for page in pages:
        content=page.page_content
        Str_pages +=content
    texts = text_splitter.split_text(Str_pages)
    embeddings=OpenAIEmbeddings()
    db=FAISS.from_texts(texts,embeddings)
    db.save_local(os.environ["INDEX_FILE"])