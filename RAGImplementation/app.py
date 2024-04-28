from flask import Flask, request, jsonify
import os
from urllib.request import urlretrieve
import numpy as np
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


app = Flask(__name__)




huggingface_embeddings = HuggingFaceBgeEmbeddings(
model_name = "BAAI/bge-small-en-v1.5",
model_kwargs={"device":"cpu"},
encode_kwargs={"normalize_embeddings":True}
)

new_vectorstore = FAISS.load_local("medical_vectorestore", huggingface_embeddings, allow_dangerous_deserialization=True)

retriever = new_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

from langchain_community.llms import HuggingFaceHub

hf=HuggingFaceHub(
repo_id="mistralai/Mistral-7B-Instruct-v0.2",
model_kwargs={"temperature":0.1, "max_length":500}
)
llm=hf

prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
2. If you find the answer, write the answer in a concise way with five sentences maximum.


{context}


Question: {question}

Helpful Answer:"""
PROMPT=PromptTemplate(
template=prompt_template, input_variables=["context", "question"]
)

retrievalQA=RetrievalQA.from_chain_type(
llm=llm,
chain_type="stuff",
retriever = retriever,
return_source_documents=True,
chain_type_kwargs={"prompt":PROMPT}
)


@app.route("/generate_answer", methods=["GET", "POST"])
def medication_generation():
    if request.method == "POST":
        query=request.json.get('query')

    result=retrievalQA.invoke({"query":query})
    answer = result['result']
    
    parts = answer.split('Helpful Answer:', 1)

    if len(parts) > 1:
        helpful_answer = parts[1].strip()
    else:
        helpful_answer = "Sorry, I couldn't find the helpful answer."
    
    return jsonify({"answer": helpful_answer})

if __name__ == "__main__":
    app.run()
            