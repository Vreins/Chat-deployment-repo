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
openai_api_key="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"
from dotenv import load_dotenv
load_dotenv()
llm=ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0)


N=0
def get_db():
    if os.path.exists(os.environ["INDEX_FILE"]):
        print("Found existing index.")
        # index = GPTVectorStoreIndex.load_from_disk(os.environ["INDEX_FILE"])
        embeddings=OpenAIEmbeddings()
        db = FAISS.load_local(os.environ["INDEX_FILE"], embeddings)
    else:
        loader=PyPDFLoader()
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
    return db
# def process_file(file):
#     try:
#         loader=PyPDFLoader(file.name)
#         pages= loader.load_and_split()
#         text_splitter = CharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         separator='\n'
#         )
#         Str_pages=''
#         for page in pages:
#             content=page.page_content
#             Str_pages +=content
#         texts = text_splitter.split_text(Str_pages)
#         embeddings=OpenAIEmbeddings()
#         global docsearch
#         docsearch = FAISS.from_texts(texts, embeddings)
#         return "Upload successful"
#     except:
#         return "Upload Unsuccessful"
# def process_url(file_path:str):
#     try:
#         file_path=str(file_path)
#         loader=PyPDFLoader(file_path)
#         pages= loader.load_and_split()
#         text_splitter = CharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         separator='\n'
#         )
#         Str_pages=''
#         for page in pages:
#             content=page.page_content
#             Str_pages +=content
#         texts = text_splitter.split_text(Str_pages)
#         embeddings=OpenAIEmbeddings()
#         global docsearch
#         docsearch = FAISS.from_texts(texts, embeddings)
#         return "Upload successful"
#     except:
#         return "Upload Unsuccessful Url not correct"

def echo(message,history):
    try:
        db=get_db()
        query=message
        docs = db.similarity_search(query)
        chain = load_qa_chain(ChatOpenAI(model="gpt-3.5-turbo"), chain_type="stuff")
        return chain.run(input_documents=docs, question=query)
    except:
        return "No file found, please upload a file or input URL"
def render_file(file):
    global N
    doc=fitz.open(file.name)
    page=doc[N]
    pix=page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    image=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
    return image
def change_input(choice):
    if choice == "Upload a file":
        return [gr.update(visible=True),gr.update(visible=False)]
    elif choice == "Enter a URL":
        return [gr.update(visible=False),gr.update(visible=True)]
html="""
<div style="text-align:center; max width: 700px;">
<h1>ChatPDF</h1>
<p> Upload a PDF, excel or csv File or input a pdf url <br> Once the document has been loaded, 
you can begin chatting with the document=)</div>
"""
css="""container{max-width:700px; margin-left: auto; margin-right:auto, padding:20px}"""

with gr.Blocks(css=css, theme=gr.themes.Monochrome()) as blocks:    
    with gr.Tab("Chat bot"):
        with gr.Column():
            chatbot=gr.ChatInterface(fn= echo,
                # label="PDF Search with GPT-4",
                title="ChatPDF",
            )
    # with gr.Tab("Upload area"):
    #     gr.HTML(html)
    #        # pdf_doc=gr.File(label="Load a pdf",file_types=['.pdf','.docx'], type='file')
    #     with gr.Column():
    #         with gr.Row():
    #             show_img=gr.Image(label='upload_pdf',tool='select', interactive=False)
    #             status=gr.Textbox(lines=1, interactive=False)
    #     pdf_file_option = gr.Radio(("Upload a file","Enter a URL"), value="Upload a file")
    #     with gr.Row(visible=True) as MainA:
    #         with gr.Row(visible=True) as colA:
    #             with gr.Row(visible=True) as rowA:
    #                 btn=gr.UploadButton("Upload a PDF", file_types=[".pdf"])
    #                 btn.upload(fn=process_file, inputs=[btn], outputs=[status]).success(fn=render_file, inputs=[btn],outputs=[show_img])
    #             with gr.Row(visible=False) as rowB:
    #                 text=gr.Textbox(lines=1, label="Input URL here")
    #                 input_btn=gr.Button("Submit URL")
    #                 input_btn.click(fn=process_url, inputs=[text], outputs=[status])
    #         pdf_file_option.change(fn=change_input, inputs=pdf_file_option, outputs=[rowA,rowB])
def createApp():
    return blocks