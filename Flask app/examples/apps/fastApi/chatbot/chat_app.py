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
llm=ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0)
# For directory processing
# def process_file(dir:str):
#     input_directory=dir
#     documents = SimpleDirectoryReader(input_dir=input_directory).load_data()
#     text_splitter = CharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
#     separator='\n'
#     )
#     Str_pages=''
#     for page in documents:
#         content=page.text
#         Str_pages +=content
#     texts = text_splitter.split_text(Str_pages)
#     embeddings=OpenAIEmbeddings()
#     vectordb = FAISS.from_texts(texts, embeddings)
#     return vectordb

# Process file function
def process(file,chain_type,k):
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
    retriever= db.as_retriever(search_type="similarity", search_kwargs={"k":k})
    qa=ConversationalRetrievalChain.from_llm(llm,
                                            chain_type=chain_type,
                                            retriever=retriever,
                                            return_source_documents=True,
                                            return_generated_question=True,)
    return qa

# Chatbot class
class cbfs(param.Parameterized):
    chat_history=param.List([])
    answer=param.String("")
    db_query=param.String("")
    db_response=param.List([])
    def __init__(self, **params):
        super(cbfs,self).__init__( **params)
        self.panels=[]
        self.loaded_file= "./chatbot/docs/McDonnell_et_al-2019-The_Obstetrician__Gynaecologist.pdf"
        self.qa= process(self.loaded_file,"stuff",4)
    
    def call_process(self,count):
        if count==0 or file_input.value is None:
            return pn.pane.Markdown(f"Loaded File:{self.loaded_file}")
        else:
            file_input.save("temp.pdf")
            self.loaded_file=file_input.filename
            button_load.button_style = "outline"
            self.qa=process("temp.pdf","stuff",4)
            button_load.button_style="solid"
        self.clr_history()
        return pn.pane.Markdown(f"Loaded file:{self.loaded_file}")
    
    def convchain(self, query):   
        if not query:
            return pn.WidgetBox(pn.Row('User:', pn.pane.Markdown("",width=600)),scroll=True)
        result= self.qa({"question":query,"chat_history":self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        self.db_query= result["generated_question"]
        self.db_response=result["source_documents"]
        self.answer=result["answer"]
        self.panels.extend([
            pn.Row('User:', pn.pane.Markdown(query, width=600)),
            pn.Row('ChatBot:', pn.pane.Markdown(self.answer, width=600 , styles={'background-color': '#F6F6F6'}))
                           ])
        inp.value = ''
        return pn.WidgetBox(*self.panels, scroll=True)
    @param.depends('db_query ', )
    def get_lquest(self):
        if not self.db_query:
            return pn.Column(
                pn.Row(pn.pane.Markdown(f"Last question to DB: ", styles={'background-color': '#F6F6F6'})),
                pn.Row(pn.pane.Str("no DB accesses so far"))
            )
        return pn.Column(
            pn.Row(pn.pane.Markdown(f"DB query:", styles={'background-color': '#F6F6F6'})),
            pn.pane.Str(self.db_query))
    @param.depends('db_response', )
    def get_sources(self):
        if not self.db_response:
            return
        rlist=[pn.Row(pn.pane.Markdown(f"Result of DB lookup:", styles={'background-color': '#F6F6F6'}))]
        for doc in self.db_response:
            rlist.append(pn.Row(pn.pane.Str(doc)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)
    @param.depends('convchain','clr_history')
    def get_chats(self):
        if not self.chat_history:
            return pn.WidgetBox(pn.Row(pn.pane.Str("No History Yet")), width=600, scroll=True)
        rlist=[pn.Row(pn.pane.Markdown(f"Current Chat History variable", styles={'background-color': '#F6F6F6'}))]
        for exchange in self.chat_history:
            rlist.append(pn.Row(pn.pane.Str(exchange)))
        return pn.WidgetBox(*rlist,width=600, scroll=True)
    def clr_history(self,count=0):
        self.chat_history=[]
        return
cb=cbfs()

file_input=pn.widgets.FileInput(accept='.pdf')
button_load=pn.widgets.Button(name="Load DB", button_type="primary")
button_clearhistory=pn.widgets.Button(name='Clear history', button_type="warning")
button_clearhistory.on_click(cb.clr_history)
inp=pn.widgets.TextInput(placeholder="Enter text here..")

bound_button_load= pn.bind(cb.call_process,button_load.param.clicks)
conversation=pn.bind(cb.convchain, inp)
tab1= pn.Column(
    pn.panel(conversation,    loading_indicator=True, height=300),
    pn.layout.Divider(),
    pn.Row(inp),
    pn.layout.Divider()
)
tab2= pn.Column(
    pn.panel(cb.get_lquest),
    pn.layout.Divider(),
    pn.panel(cb.get_sources ),
)
tab3= pn.Column(pn.panel(cb.get_chats),
                pn.layout.Divider(),)
tab4= pn.Column(
    pn.Row( file_input, button_load, bound_button_load),
    pn.Row( button_clearhistory, pn.pane.Markdown("Clears chat history. Can use to start new topic")),)

dashboard= pn.Column(
    pn.Row(pn.pane.Markdown('# ChatWithYourData_Bot')),
    pn.Tabs(('Conversation',tab1), ('Database', tab2), ('Chat History',tab3),('Configure',tab4))
)
pn.extension

def createApp():
    return dashboard.servable()