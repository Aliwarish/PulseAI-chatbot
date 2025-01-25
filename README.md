# PulseAI-chatbot
PulseAI: The LLM Core for Real-Time Global News 

# Chatbot Application: PDF and Google Search Interfaces

This repository contains a Streamlit-based chatbot application that offers two primary modes of operation:
1. **PDF Chatbot**: Answer questions based on the content of uploaded PDF files.
2. **Google Search Chatbot**: Fetch and provide answers from Google search results.

## Features

### PDF Chatbot
- Upload PDF files and interact with the content using natural language queries.
- Extracts and indexes PDF content into a Pinecone vector store for fast retrieval.
- Provides answers based on indexed documents using LangChain's question-answering chains.

### Google Search Chatbot
- Fetches search results using the Google Serper API.
- Uses context from previous interactions to provide accurate and relevant responses.
- Incorporates conversation history for better user experience.

## Technologies Used
- **Python**
- **Streamlit**: For the frontend interface.
- **LangChain**: For managing embeddings, document creation, and question-answering chains.
- **Pinecone**: As a vector database for efficient document retrieval.
- **Google Serper API**: For fetching search results from Google.
- **Fitz (PyMuPDF)**: For PDF text extraction.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add API Keys**:
   - Create a `together_key.yml` file in the root directory.
   - Add the following keys:
     ```yaml
     TOGETHER_API_KEY: "<your_together_api_key>"
     PINECONE_API_KEY: "<your_pinecone_api_key>"
     SERPER_API_KEY: "<your_serper_api_key>"
     ```

## How to Run

1. **Start the Application**:
   ```bash
   streamlit run app.py
   ```

2. **Select Chatbot Mode**:
   - Use the sidebar to switch between **PDF Chatbot** and **Google Search Chatbot**.

### PDF Chatbot Workflow
- Upload a PDF file.
- Ask a question based on the content of the file.
- View the chatbot’s response and conversation history.

### Google Search Chatbot Workflow
- Enter a query in the text input.
- Get responses from the Google Serper API.
- View conversation history.

## File Structure
- `app.py`: Main application script.
- `together_key.yml`: Contains API keys for external services (not included in the repository for security).
- `requirements.txt`: List of dependencies required to run the project.

## Key Functions

### Helper Functions
- `extract_text_from_pdf(file_path)`: Extracts text from a PDF file.
- `create_documents(file_path)`: Splits PDF text into chunks for indexing.
- `generate_index_name(file_name)`: Creates a valid index name for Pinecone.

### Backend Functions
- `create_or_get_pinecone_index(index_name)`: Ensures the Pinecone index exists.
- `index_pdf_in_pinecone(file_path)`: Indexes PDF content into Pinecone.
- `query_pinecone(user_query)`: Queries the Pinecone index for relevant documents.
- `answer_pdf_query(user_query)`: Provides answers based on the indexed PDF content.
- `fetch_google_search_results(user_query)`: Fetches search results using the Google Serper API.
- `answer_google_query(user_query)`: Provides answers based on Google search results.

### Streamlit UI
- `pdf_chat_interface()`: Interface for PDF-based chatbot.
- `google_chat_interface()`: Interface for Google search chatbot.
- `chatbot_application()`: Main application entry point with mode selection.

## Future Enhancements
- Add support for other document formats (e.g., Word, Excel).
- Improve UI/UX for better user interaction.
- Enhance error handling and logging for better debugging.
- Incorporate more advanced conversational AI models.

## License
This project is licensed under the [MIT License](LICENSE).



---


