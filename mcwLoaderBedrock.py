import boto3
import json
import os

from astrapy.constants import Environment
from langchain_astradb import AstraDBVectorStore
from langchain_community.vectorstores import Cassandra
from langchain_aws import BedrockEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# env vars
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY= os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION="us-west-2"

ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT= os.environ.get("ASTRA_DB_API_ENDPOINT")

TABLE_NAME = "minecraft_vectors_bedrock"

# using Amazon Bedrock
awsSession = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

awsClient = awsSession.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
)

# Test boto session w/ S3
#s3Client = awsSession.client(
#	"s3",
#	region_name=AWS_REGION)
#
#print(s3Client.list_buckets())
#
# ^^ works! ^^

# Bedrock embeddings
embeddings = BedrockEmbeddings(
    credentials_profile_name=None,
    client=awsClient,
    model_id="amazon.titan-embed-text-v1",
    endpoint_url=None,
    region_name=AWS_REGION,
)

#embedded_sentence = embeddings.embed_query("This is a sample sentence")
#print(embedded_sentence)

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
