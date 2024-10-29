import boto3
import json
import os

from astrapy.constants import Environment
from langchain_astradb import AstraDBVectorStore
from langchain_aws import BedrockEmbeddings
#from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# env vars
ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
SECRET_KEY= os.environ.get("AWS_SECRET_ACCESS_KEY")
ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT= os.environ.get("ASTRA_DB_API_ENDPOINT")
TABLE_NAME = "minecraft_vectors_bedrock"

# using Amazon Bedrock
awsSession = boto3.Session(
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
)

awsClient = awsSession.client(
    "bedrock-runtime",
    region_name="us-west-2",
    endpoint_url="https://bedrock.us-west-2.amazonaws.com"
)

#model_kwargs = {
#	"normalize": False,
#   "embedding_dimensions": 1536
#}

# Bedrock embeddings
embeddings = BedrockEmbeddings(
	client=awsClient,
	region_name="us-west-2"
	model_id="amazon.titan-embed-text-v1",
#	model_kwargs=model_kwargs
)

# OpenAI embeddings
#OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
#embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)


# testing TOKEN
#print("token=" + ASTRA_DB_APPLICATION_TOKEN)

# DB connection
vectorstore = AstraDBVectorStore(
    embedding=embeddings,
    namespace="default_namespace",
    collection_name=TABLE_NAME,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    environment=Environment.DSE
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 10,
    length_function = len,
    is_separator_regex = False,
)

linecounter = 0

# iterate through all (five) of the Minecraft text files
for counter in range(1,6):
#for counter in range(4,5):
#for counter in range(4,6):
	textfile = str(counter) + ".txt"

	docs = []
	with open(textfile) as ft:
		doc = ft.read()
		texts = text_splitter.split_text(doc)

		for index in range(0,len(texts)):
			text = str(texts[index]).strip().replace("\n"," ").replace("\"","\\\"")
			emb = embeddings.embed_query(text)

			document = Document(page_content=text);

			docs.append(document)
			linecounter = linecounter + 1

	vectorstore.add_documents(docs)
	print(f"{textfile} successfully processed.")
