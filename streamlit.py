import os
import streamlit as st
import pickle
import time

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS


from dotenv import load_dotenv
load_dotenv() 

st.title("News Research Tool 📈" )

st.sidebar.title("News Article URLs")

urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)


process_url_clicked = st.sidebar.button("Process URLs")

file_path = "faiss_index"

main_placeholder = st.empty()

llm = ChatGoogleGenerativeAI(
    model = "models/gemini-3.1-flash-lite",
    temperature=0.9,
    max_tokens=500,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("data Loading Started... ✅")
    data = loader.load()

    ### split data

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )

    main_placeholder.text("Text Splitter Started✅")
    docs = text_splitter.split_documents(data)

    # create embeddings and save it to FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vectorstore_gemini = FAISS.from_documents(docs, embeddings)

    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
  
    # Save the FAISS index locally
    time.sleep(2)
    
    vectorstore_gemini.save_local(file_path)
    
    main_placeholder.text("FAISS index saved successfully! ✅")
    
query = st.text_input("Question:")

if query:
    if os.path.exists(file_path):
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        vectorestore = FAISS.load_local(
            file_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorestore.as_retriever()
        )

        result = chain(
            {"question": query},
            return_only_outputs=True
        )

        st.header("Answer")
        st.write(result["answer"])

        if result.get("sources"):
            st.subheader("Sources")
            st.write(result["sources"])
    else:
        st.warning("Please process the URLs first.")
