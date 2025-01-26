from google.cloud import firestore
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_google_firestore import FirestoreChatMessageHistory
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

# Firestore Configuration
PROJECT_ID = 'langchainfirestore'
SESSION_ID = 'user_session_new'
COLLECTION_NAME = 'chat_history'

print('Initializing Firestore client...')
client = firestore.Client(project=PROJECT_ID)

print('Initializing Firestore message history...')
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)

print('Chat history initialized.')
print('Current chat history:')
for message in chat_history.messages:
    print(f" Content: {message}")


callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

print('Starting chat with AI...')
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.7,
    max_tokens=200,
    stop=["\nHuman:"],
    callbacks=callback_manager,
    verbose=False,
    streaming=True,
)

# Chat Loop
while True:
    try:
        # Get user input
        print()
        human_input = input('Enter a message: ').strip()
        if human_input.lower() == 'quit':
            print("Goodbye!")
            break

        # Add user input to Firestore
        chat_history.add_user_message(human_input)

        # Trim chat history to fit within the context window
        chat_history.messages = chat_history.messages

        # Generate AI Response
        ai_response = llm.invoke(chat_history.messages).strip()

        # Post-process response to remove repetition
        ai_response = ai_response.split("\n")[0]  # Keep only the first line
        if not chat_history.messages or ai_response != chat_history.messages[-1].content:
            chat_history.add_ai_message(ai_response)

    except Exception as e:
        print(f"Error: {e}")