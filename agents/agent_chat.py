from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
import os
from langchain_community.llms import LlamaCpp
import wikipedia

# Load environment variables
load_dotenv()

# Get model path
model_path = os.getenv("MODEL_PATH")
if not model_path:
    raise ValueError("MODEL_PATH is not set in the .env file or environment variables.")

# Initialize LLM with a smaller max_tokens to avoid context overflow
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.3,
    max_tokens=100,  # Further reduced for safety
    verbose=False,
    streaming=True,
)

# Wikipedia search function with improved error handling
def search_wikipedia(query):
    """Fetches a brief Wikipedia summary with error handling."""
    try:
        summary = wikipedia.summary(query, sentences=2)
        return f"Wikipedia Summary: {summary}"
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple topics found: {', '.join(e.options[:3])}"
    except wikipedia.exceptions.PageError:
        return "No relevant Wikipedia page found."
    except Exception as e:
        return f"Wikipedia error: {str(e)}"

# Define tools with structured responses
tools = [
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Use this tool to search for information on Wikipedia.",
    )
]

# Pull structured chat agent prompt from LangChain Hub
prompt = hub.pull("hwchase17/structured-chat-agent")

# Initialize conversation memory with a token limit
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, max_token_limit=350
)

# Create structured chat agent
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

# Create an AgentExecutor with better error handling
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

# Initial system message
initial_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the Wikipedia tool."
memory.chat_memory.add_message(SystemMessage(content=initial_message))

# Chat loop
while True:
    try:
        user_input = input("User: ").strip()
        if user_input.lower() == "exit":
            print("Exiting chat.")
            break
        
        # Add user's message to conversation memory
        memory.chat_memory.add_message(HumanMessage(content=user_input))
        
        # Invoke agent executor with user input
        response = agent_executor.invoke({"input": user_input})
        output_text = response.get("output", "I could not generate a valid response.")

        # Ensure valid response before storing in memory
        if output_text and "Invalid or incomplete response" not in output_text:
            print("Bot:", output_text)
            memory.chat_memory.add_message(AIMessage(content=output_text))
        else:
            print("Bot: Sorry, I couldn't find an answer.")
    except Exception as e:
        print(f"Error: {str(e)}")