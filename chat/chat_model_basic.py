from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
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

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
template = """
Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer.
"""

prompt = PromptTemplate.from_template(template)

# Initialize the LLM 
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.75,
    max_tokens=13000,
    top_p=1,
   callbacks=callback_manager, 
   verbose=False,
)  
llm_chain = prompt | llm
question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"
llm_chain.invoke({"question": question})


