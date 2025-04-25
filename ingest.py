from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load and split PDF
loader = PyPDFLoader("data/insurance_policy_info.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Use Sentence Transformers (all-MiniLM-L6-v2)
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store in Chroma vectorstore
db = Chroma.from_documents(docs, embedding, persist_directory="vectorstore")
db.persist()
