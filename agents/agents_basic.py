from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain_community.llms import LlamaCpp
from langchain.agents.output_parsers import ReActSingleInputOutputParser
import datetime
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

llm = LlamaCpp(
    model_path=model_path,
    temperature=0.3,  # Lower temperature for deterministic output
    max_tokens=200,
    verbose=False,
    streaming=True,
)

# Define the Time tool
def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    try:
        now = datetime.datetime.now()
        time_str = now.strftime("%I:%M %p")
        print(f"[DEBUG] Time tool called. Current time: {time_str}")
        return f"The current time is {time_str}."
    except Exception as e:
        print(f"[ERROR] Error in Time tool: {e}")
        return "Error: Unable to fetch the current time."

# Define tools
tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Use this tool to fetch the current time.",
    )
]

# Define the ReAct prompt template
react_prompt_template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

# Create a PromptTemplate
prompt_template = PromptTemplate(
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
    template=react_prompt_template,
)

# Create the ReAct agent
output_parser = ReActSingleInputOutputParser()

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_template,
    output_parser=output_parser,
    stop_sequence=["\nObservation"],  # Explicit stop sequence
)

# Create an AgentExecutor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

# Run the agent
try:
    print("[DEBUG] Invoking agent...")
    response = agent_executor.invoke({"input": "What time is it?"})
    print("[DEBUG] Agent response:", response)
except Exception as e:
    print(f"[ERROR] Agent execution failed: {e}")
    response = None

# Print final response
if response:
    print("Final Output:", response.get("output", "No output received."))
else:
    print("[ERROR] No valid response from the agent.")