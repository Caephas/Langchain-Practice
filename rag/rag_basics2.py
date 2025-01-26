import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

# Ensure the persistent directory exists or create it if necessary
if not os.path.exists(persistent_directory):
    print("Persistent directory not found, creating a new vector store.")

    # Load the document
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split the text into chunks
    #The RecursiveCharacterTextSplitter gives something more meaningful
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Initialize the embedding model
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings()

    # Create and persist the vector store
    print("Creating Chroma vector store...")
    db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persistent_directory)
    db.persist()
    print("Finished creating and persisting the Chroma vector store.")

else:
    print("Using existing Chroma vector store.")
    embeddings = HuggingFaceEmbeddings()
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Define the query
query = "How to test prototype with target users"

# Retrieve relevant documents, the thresholds are important in document retrieval
retriever = db.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'k': 5, 'score_threshold': 0.4},
)
relevant_docs = retriever.get_relevant_documents(query)

# Display the results
print("\nRelevant documents:")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}: \n{doc.page_content}\n")
    if doc.metadata:
        print(f"Document metadata:\n{doc.metadata.get('source', 'unknown')}")