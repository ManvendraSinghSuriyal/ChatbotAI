import os
import openai
from bs4 import BeautifulSoup
from langchain_community.document_loaders import ConfluenceLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from dotenv import load_dotenv, find_dotenv
import requests
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

pc = Pinecone(api_key="118165d4-829b-4bd6-aa4c-0c3be9cdde9d")
index = pc.Index("my-movies")


loader = ConfluenceLoader(
    url="https://nagarrochatbot.atlassian.net",
    username="abhijeet12jats@icloud.com",
    api_key="ATATT3xFfGF0-R1UqebEFlmFk3DAv3z1nPdZYLwpxH0AnApq_kCud9e5QW-0Amq2ytwU84K5VphYKwJT8GF3edr-W6hxQJZ5eZWK62wOlBeZYoj48hlw2dszO68k-DE4IsRFfwXi-S9YN-WeTgoPxYzUOFYezhhtmOaz_ozmJfGZTK5hL8guzqI=17A2C431",
    space_key="~7120200c85513f7f1e47a6b8698864d8e3e77a",
    include_attachments=True,
    limit=50
)
documents = loader.load()


newData = []
for document in documents:
    newData.append(document.page_content)
    newData.append(document.page_content)
    newData.append(document.page_content)

newData.pop()

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newData).toarray()
vectors_list = [vec.tolist() for vec in vectors]

newList = []
i = 0
for vec in vectors_list:
    i = i+1
    newList.append({"id": str(i),"values":vec})

# print(newList)
index.upsert(vectors=newList,
    namespace= "ns1")
#     })
#     print("Document Title:", document.page_content)
#     print("------------------------")

print(index.query(
    namespace="ns1",
    id="1",
    top_k=2,
))
