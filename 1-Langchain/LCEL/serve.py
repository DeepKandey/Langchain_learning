from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes

import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="gemma2-9b-it",groq_api_key=groq_api_key)

# 1. create prompt template
system_template = "Translate the following from English to {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', system_template),
        ('user','{text}')
    ]
)

parser = StrOutputParser()

# app definition
app = FastAPI(title="Langchain Server",
              version=1.0,
              description="A simple API server using Langchain runnable interfaces")

# chain create
chain = prompt_template|model|parser

# adding chain routes
add_routes(
    app = app,
    runnable = chain,
    path = "/chain"
)

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host = "localhost",port=8000)