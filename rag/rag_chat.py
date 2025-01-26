import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.llamacpp import LlamaCpp
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_core.callbacks import CallbackManager
from langchain_core.prompts import ChatPromptTemplate
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
persistent_directory = os.path.join(current_dir, "database", "chroma_db")

# Callback Manager
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Initialize LlamaCpp model
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.8,
    max_tokens=2048,
    n_ctx=2048,
    top_p=0.88,
    echo=False,
    callbacks=callback_manager,
    verbose=False,
    streaming=True,
    stop=["Q:", "\nHuman:"]
)
# Initialize embeddings and Chroma vector store
embeddings = HuggingFaceEmbeddings()
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)

# Query for retrieving relevant documents
query = "How to test prototype with target users"

# Retrieve relevant documents
retriever = db.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 1}  # Retrieve top 1 document
)
relevant_docs = retriever.get_relevant_documents(query)

# Combine relevant document content into a single string
documents_content = "\n\n".join([doc.page_content for doc in relevant_docs])

# Define the prompt structure using ChatPromptTemplate
messages = [
    ('system', "You are a helpful assistant. Use only the provided documents to answer questions."),
    ('user', "Given the following documents:\n\n{documents}\n\nAnswer this question: {query}")
]

# Create the ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_messages(messages)

# Format the prompt
formatted_prompt = prompt_template.format(
    documents=documents_content,
    query=query
)

# Send the formatted prompt to the LLM
print("\n------ Generated Response -----")
response = llm.invoke([formatted_prompt])  # Pass the formatted prompt as a list
