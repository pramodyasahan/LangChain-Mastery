from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

# Setup environment variables and messages
load_dotenv()

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 75 divided by 5?"),
]

model = ChatOpenAI(model="gpt-4o")

result = model.invoke(messages)
print(f"Answer from OpenAI: {result.content}")

model = ChatAnthropic(model="claude-3")
result = model.invoke(messages)
print(f"Answer from Anthropic: {result.content}")

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
result = model.invoke(messages)
print(f"Answer from Google: {result.content}")
