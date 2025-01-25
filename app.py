import os
import locale
import streamlit as st
import asyncio
import warnings
import yaml
import fitz  # For PDF processing
import re
import tempfile
from pinecone import Index, create_index, list_indexes
from langchain import GoogleSerperAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_together import ChatTogether
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import HumanMessage

# Ignore warnings
warnings.filterwarnings("ignore")

# Configure locale
locale.getpreferredencoding = lambda: "UTF-8"

# Load API credentials
with open("together_key.yml", 'r') as file:
    api_creds = yaml.safe_load(file)

os.environ["TOGETHER_API_KEY"] = api_creds.get('TOGETHER_API_KEY', "")
os.environ["PINECONE_API_KEY"] = api_creds.get('PINECONE_API_KEY', "")
os.environ["SERPER_API_KEY"] = api_creds.get('SERPER_API_KEY', "")

# Initialize Pinecone embeddings
async def initialize_embeddings():
    return PineconeEmbeddings(
        model='multilingual-e5-large',
        pinecone_api_key=os.environ.get('PINECONE_API_KEY')
    )

embeddings = asyncio.run(initialize_embeddings())

### ---------------- HELPER FUNCTIONS ---------------- ###
def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    reader = fitz.open(file_path)
    text = ""
    for page in reader:
        text += page.get_text()
    return text

def create_documents(file_path):
    """Create documents from the text extracted from a PDF."""
    text = extract_text_from_pdf(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len)
    return text_splitter.create_documents([text])

def generate_index_name(file_name):
    """Generate a valid index name from the file name."""
    raw_index_name = file_name.split(".")[0][:7].lower()
    return re.sub(r"[^a-z0-9-]", "-", raw_index_name).strip("-").replace('"', '').replace("'", "")

### ---------------- BACKEND FUNCTIONS ---------------- ###
def create_or_get_pinecone_index(index_name, dimension=1536, metric="cosine"):
    """Ensure the Pinecone index exists, or create it if not."""
    existing_indexes = list_indexes()
    if index_name not in existing_indexes:
        create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
        )
        st.info(f"Index '{index_name}' created.")
    return Index(index_name)

def index_pdf_in_pinecone(file_path):
    """Index PDF content into the Pinecone index."""
    index_name = "ragreader"
    documents = create_documents(file_path)
    PineconeVectorStore.from_documents(
        documents=documents,
        index_name=index_name,
        embedding=embeddings,
        namespace="ragreader_namespace"
    )

def query_pinecone(user_query):
    """Query the Pinecone index for relevant documents."""
    index_name = "ragreader"
    namespace = "ragreader_namespace"
    docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=namespace)
    return docsearch.similarity_search(user_query, k=3)

def answer_pdf_query(user_query):
    """Answer questions using the Pinecone index with conversation history."""
    if "pdf_conversation_history" not in st.session_state:
        st.session_state.pdf_conversation_history = []

    llm = ChatTogether(
        openai_api_key=os.environ.get('TOGETHER_API_KEY'),
        model_name='meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo',
        temperature=0.0
    )
    chain = load_qa_chain(llm)

    documents = query_pinecone(user_query)

    conversation_context = " ".join([f"User: {msg['user']} Bot: {msg['bot']}" 
                                      for msg in st.session_state.pdf_conversation_history])

    prompt = f"""
    Context from previous conversation:
    {conversation_context}

    Current question:
    {user_query}

    Provide a concise and accurate response based on the context and the provided documents.
    """
    response = chain.run(input_documents=documents, question=prompt)

    st.session_state.pdf_conversation_history.append({"user": user_query, "bot": response})
    return response

def fetch_google_search_results(user_query):
    """Fetch search results using Google Serper API."""
    search = GoogleSerperAPIWrapper(gl="in", k=5)
    return search.run(user_query)

def answer_google_query(user_query):
    """Provide answers based on Google search results with conversation history."""
    if "google_conversation_history" not in st.session_state:
        st.session_state.google_conversation_history = []

    context = fetch_google_search_results(user_query)

    conversation_context = " ".join([f"User: {msg['user']} Bot: {msg['bot']}" 
                                      for msg in st.session_state.google_conversation_history])

    prompt = f"""
    Context from previous conversation:
    {conversation_context}

    Current question:
    {user_query}

    Use the following context from Google Search results to provide an accurate answer:
    {context}
    """
    client = ChatTogether(model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo")
    message = [HumanMessage(content=prompt)]
    response = client.invoke(message).content

    st.session_state.google_conversation_history.append({"user": user_query, "bot": response})
    return response

### ---------------- STREAMLIT UI ---------------- ###
def pdf_chat_interface():
    st.title("PDF-based Chatbot")
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

    if "pdf_conversation_history" not in st.session_state:
        st.session_state.pdf_conversation_history = []

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name

        index_pdf_in_pinecone(temp_file_path)

        user_query = st.text_input("Ask a question:")
        if user_query:
            response = answer_pdf_query(user_query)
            st.write("### Chat Conversation:")
            for message in st.session_state.pdf_conversation_history:
                st.write(f"**You:** {message['user']}")
                st.write(f"**Bot:** {message['bot']}")

def google_chat_interface():
    st.title("Google Search-based Chatbot")
    if "google_conversation_history" not in st.session_state:
        st.session_state.google_conversation_history = []

    user_query = st.text_input("Enter your query:")
    if user_query:
        response = answer_google_query(user_query)
        st.write("### Chat Conversation:")
        for message in st.session_state.google_conversation_history:
            st.write(f"**You:** {message['user']}")
            st.write(f"**Bot:** {message['bot']}")

def chatbot_application():
    st.sidebar.title("Choose Chatbot Mode")
    option = st.sidebar.selectbox("Chatbot Mode:", ["PDF Chatbot", "Google Search Chatbot"])

    if option == "PDF Chatbot":
        pdf_chat_interface()
    elif option == "Google Search Chatbot":
        google_chat_interface()

if __name__ == "__main__":
    chatbot_application()
