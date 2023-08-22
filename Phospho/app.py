import os
import pinecone
import openai
from dotenv import load_dotenv

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader

# Load env variables from .env file
load_dotenv()

# Get env variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')

# Connect to Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
openai.api_key = OPENAI_API_KEY


loader = TextLoader('tmp/text.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Setting up embeddings provider
embeddings = OpenAIEmbeddings()

# Generate embeddings using OpenAI and store them in the Pinecone index
docsearch = Pinecone.from_documents(docs, embeddings, index_name=PINECONE_INDEX_NAME)
