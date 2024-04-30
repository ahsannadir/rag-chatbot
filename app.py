# Importing Libraries
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceHubEmbeddings

# Extracting raw texts from the PDF files.
def get_text_from_pdf(pdf) -> str:
    text = ""
    for p in pdf:
        pdf_reader = PdfReader(p)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Processing & Converting the texts into chunks
def get_chunks(text: str) -> str:
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Getting Vector Database and Embeddings
def get_vectors(chunks: str):
    embeddings = HuggingFaceHubEmbeddings(model = "avsolatorio/GIST-Embedding-v0", huggingfacehub_api_token = "hf_token_here")
    vectors = FAISS.from_texts(texts = chunks, embedding = embeddings)
    
    return vectors

# Getting Conversational Chain 
def get_conversational_chain(vectors):
    llm = HuggingFaceHub(repo_id = 'google/flan-t5-large', model_kwargs = {"temperature": 0.9, "max_length": 2048})

    memory = ConversationBufferMemory(
        memory_key = 'chat_history', return_messages = True
    )

    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectors.as_retriever(),
        memory = memory
    )
    
    return conversational_chain

# Main Function
def main():
    st.set_page_config(
    page_title = 'RAG Chatbot',
    page_icon = "🤖",
    )

    st.title("RAG Chatbot 🤖", anchor = None)

    load_dotenv()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    

    pdf_files = st.file_uploader("Upload a PDF", type = "pdf", accept_multiple_files = True, label_visibility = 'hidden')

    if pdf_files:
        with st.spinner("Processig the document"):
            raw_text = get_text_from_pdf(pdf_files)
            text_chunks = get_chunks(raw_text)
            vector = get_vectors(text_chunks)
            st.session_state.conversation = get_conversational_chain(vector)


    question = st.chat_input("Ask anything")

    if question:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner("Thinking"):
            response = st.session_state.conversation({'question': question})
            st.session_state.chat_history = response['chat_history']

            st.session_state.chat_history.append({
                "role": "Assistant",
                "content": response['answer']
            })

            with st.chat_message("Assistant"):
                st.markdown(response['answer'])


if __name__ == '__main__':
    main()