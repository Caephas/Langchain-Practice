from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.runnables import RunnableLambda, RunnableSequence
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
messages = [
    ('system', "You are a system design expert."),
    ('user', "Tell me how best {design_question}.")
]

# Create the ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_messages(messages)

# Define the chain components
format_prompt = RunnableLambda(lambda x: prompt_template.format(**x))
invoke_model = RunnableLambda(lambda x: llm.invoke([x]))
parse_output = RunnableLambda(lambda x: x[0])  # Assuming LlamaCpp outputs a list of responses

# Build the chain
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# Invoke the chain
response = chain.invoke({"design_question": "to build a website"})
