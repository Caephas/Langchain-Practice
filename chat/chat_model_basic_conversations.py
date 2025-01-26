from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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


# Callback manager for streaming output
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Initialize the LLM
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.3,
    max_tokens=200,
    callbacks=callback_manager,
    verbose=False,
    streaming=True
)
chat_history = []
# Initial messages
system_message = SystemMessage(content="You are a knowledgeable assistant specializing in Real Madrid. Answer questions succinctly and wait for further input before continuing.")
chat_history.append(system_message)

# Conversation loop
while True:
    # Get user input
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    chat_history.append(HumanMessage(content=user_input))

    # Generate the AI's response
    try:
        # Limit chat history to the last 4 exchanges (SystemMessage + 3 most recent turns)
        recent_history = chat_history[-5:] if len(chat_history) > 5 else chat_history

        response = llm.invoke(recent_history).strip()
        if response:
            # Check if the response is not a duplicate
            if not chat_history or response != chat_history[-1].content:
                print(f"AI: {response}")
                chat_history.append(AIMessage(content=response))
            else:
                print("AI: (No new information)")
        else:
            print("AI: (No response)")
    except Exception as e:
        print(f"An error occurred: {e}")
        break

print('----------------Message History--------------')
print(chat_history)