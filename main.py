from langchain.chat_models import ChatOpenAI
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,MessagesPlaceholder)
from langchain.agents import OpenAIFunctionsAgent,AgentExecutor
from langchain.schema import SystemMessage
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from tools.sql import run_query_tool,list_tables,describe_tables_tool
from tools.report import write_report_tool
from handlers.chat_model_start_handler import ChatModelStartHandler

load_dotenv()
handler = ChatModelStartHandler()
chat = ChatOpenAI(
    callbacks = [handler, handler,handler]
)
tables = list_tables()
prompt = ChatPromptTemplate(messages = [
    SystemMessage(content = 
                  "You are an ai that has an access to a SQLite database."
                  f"The database has tables of : {tables} \n"
                  "Do not make assumptions about what tables exist"
                  "or what column exists. Instead, use the 'describe_tables' function"),
    MessagesPlaceholder(variable_name = "chat_history"),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder(variable_name = "agent_scratchpad")
])

memory = ConversationBufferMemory(memory_key = "chat_history",return_messages = True)

tools =[run_query_tool,describe_tables_tool,write_report_tool]

agent = OpenAIFunctionsAgent(
    llm = chat,
    prompt = prompt,
    tools =tools
)
agent_executor = AgentExecutor(
    agent = agent,
    verbose = True, 
    tools = tools,
    memory = memory
)
agent_executor("How many orders are there ? Write the results to a report file. ")

agent_executor("Repeat the same process for the users.")