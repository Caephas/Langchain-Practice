from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
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

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path=model_path,
    temperature=0.5,
    max_tokens=500,
    callbacks=callback_manager,
    verbose=False,
    streaming=True
)

# Define messages for the ChatPromptTemplate
messages =  [
    ('system', "You are a system design expert"),
    ('user', "Tell me how best {design_question}.")
]

# Create the ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_messages(messages)

# Format the prompt
formatted_prompt = prompt_template.format(
    design_question="to build a RAG system for my portfolio"
)

# Send the formatted prompt to the LLM
response = llm.invoke([formatted_prompt])  # Pass the formatted prompt as a list