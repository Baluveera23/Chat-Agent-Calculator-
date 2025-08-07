import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool, AgentType
import numexpr

# Check for Google API key
os.environ["GOOGLE_API_KEY"] = st.text_input("Enter your Google API Key:", type="password")

# Initialize LLM and agent as before
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.8,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)


def calculator_tool(expression: str) -> str:
    try:
        result = numexpr.evaluate(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

calculator = Tool(
    name="Calculator",
    func=calculator_tool,
    description="Perform math calculations given mathematical expressions, like '2+2*3'."
)

tools = [calculator]

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
)

st.title("ChatAgent Calculator🧮")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

def process_input():
    user_input = st.session_state["input_box"]
    if user_input.strip():
        # Add user input to messages
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Get response from agent
        with st.spinner("Generating....."):
            response = agent_executor.run(user_input)
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        # Clear input box by resetting state value
        st.session_state["input_box"] = ""

# Display the conversation history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"**❓:** {msg['content']}")
    else:
        st.markdown(f"**🟰 :** {msg['content']}")
        st.write("-----------------------------------------------------------------------------------------------------")
# Single input box with callback to process input and clear box after submission
st.text_input(
    "Enter your question or calculation:",
    key="input_box",
    on_change=process_input,
    placeholder="Type your question here and press Enter..."
)