import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the local LLM
model_path = os.getenv("MODEL_PATH")

# Check if the model path is loaded correctly
if not model_path:
    raise ValueError("MODEL_PATH is not set in the .env file or environment variables.")

print(f"Model path loaded: {model_path}")

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "inspired.pdf")
persistent_directory = os.path.join(current_dir, "database", "chroma_db")


# Check if the vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory not found, creating it.")

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load the PDF document
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split the text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    print("\nDocument chunks information:")
    print(f"Number of chunks: {len(docs)}")
    print(f"Sample chunk: {docs[0].page_content}")

    # Create embeddings
    print("\nCreating embeddings...")
    # Define the embedding model
    embedding = HuggingFaceEmbeddings()
    print("Finished creating embeddings.")

    # Create and persist the vector store
    print("\nCreating vector store...")
    db = Chroma.from_documents(docs, embedding, persist_directory=persistent_directory)
    print("Finished creating vector store.")
else:
    print("Vector store already exists.")