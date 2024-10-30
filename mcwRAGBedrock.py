import boto3
import os

from astrapy.constants import Environment
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_astradb import AstraDBVectorStore
from langchain_aws import BedrockEmbeddings, ChatBedrock

# define env vars
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY= os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION="us-west-2"

ASTRA_DB_API_ENDPOINT= os.environ.get("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.environ.get('ASTRA_DB_APPLICATION_TOKEN')
CASSANDRA_KEYSPACE = "default_namespace"
TABLE_NAME = "minecraft_vectors_bedrock"

minecraft_assistant_template = """
You are an assistant for the game Minecraft, helping players with questions.
Answer the questions with the context provided, but you may use external sources as well.
You must refuse to answer any questions not related to the game Minecraft.

CONTEXT:
{context}

QUESTION: {question}

YOUR ANSWER:"""
minecraft_prompt = ChatPromptTemplate.from_template(minecraft_assistant_template)

# using Amazon Bedrock
awsSession = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

awsClient = awsSession.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
)

# Bedrock chat
llm = ChatBedrock(
    client=awsClient,
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    region_name=AWS_REGION
)

# Bedrock embeddings
embeddings = BedrockEmbeddings(
    credentials_profile_name=None,
    client=awsClient,
    model_id="amazon.titan-embed-text-v1",
    endpoint_url=None,
    region_name=AWS_REGION,
)

vectorstore = AstraDBVectorStore(
    embedding=embeddings,
    namespace=CASSANDRA_KEYSPACE,
    collection_name=TABLE_NAME,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    environment=Environment.DSE
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | minecraft_prompt
    | llm
    | StrOutputParser()
)

userInput = "What is the recipe for an iron helmet?"

print(f"Question? {userInput}")

while userInput != "exit":
    print(chain.invoke(userInput))
    print("\n")
    userInput = input("Next question? ")

print("Exiting...")
