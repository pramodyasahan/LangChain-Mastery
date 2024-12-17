from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Loading the model
model = ChatOpenAI(model="gpt-3.5-turbo")

messages = [
    SystemMessage(content="Solve the following math problem"),
    HumanMessage(content="What are the days in a week")
]

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from GPT: {result.content}")

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 5 times 5?"),
    AIMessage(content="5 times 5 which gives the value of 25"),
    HumanMessage(content="What is 100 divide by 5?")
]

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from GPT: {result.content}")
