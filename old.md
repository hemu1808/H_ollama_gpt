# HemGPT1

This project is a Retrieval-Augmented Generation (RAG) system that enables efficient document querying and answering using ChromaDB for vector storage, Ollama for generating answers based on context, and Streamlit for the user interface. The system supports document upload (PDF format), semantic search, and question-answering using document chunks.

## Features

- **Document Upload**: Upload PDF documents to be indexed.
- **Semantic Search**: Query the indexed documents and retrieve relevant results.
- **Contextual Question Answering**: Generate answers using an AI model (Ollama) based on the retrieved context.
- **Document Re-ranking**: Use a cross-encoder model to re-rank the retrieved documents for better relevance.
- **Streamlit Interface**: A simple and intuitive UI for uploading documents and asking questions.

## Tech Stack

- **Python**: Main programming language for building the application.
- **Streamlit**: Web framework for the user interface.
- **ChromaDB**: A database for storing document embeddings.
- **Ollama**: API used for generating answers based on context.
- **PyMuPDF**: Used to extract content from PDF documents.
- **Langchain**: Framework for chaining together document loaders, text splitters, and language models.
- **Sentence-Transformers**: Model for re-ranking documents using a cross-encoder.

## Requirements

Here are the required Python packages:

- `chromadb`
- `ollama`
- `streamlit`
- `langchain_community`
- `sentence-transformers`
- `PyMuPDF`
- `langchain_core`

## Usage

1. **Upload PDF Document**: In the sidebar, click the file uploader to upload a PDF file. The document will be processed and split into smaller chunks.
2. **Ask Questions**: Enter your question in the text box provided and click "Submit". The system will retrieve relevant document chunks and generate an answer.
3. **See Retrieved Documents**: Click on the "See retrieved documents" expander to view the documents retrieved for your query.
4. **See Document IDs**: Click on the "See most relevant document ids" expander to view the IDs of the most relevant documents used to generate the response.

## How it Works

1. **Document Processing**: The PDF file is split into smaller chunks, which are then embedded into a vector space using an embedding function.
2. **Vector Storage**: These document embeddings are stored in a ChromaDB collection for efficient querying.
3. **Querying**: When a user submits a query, the system performs a semantic search by comparing the query to the stored embeddings.
4. **Document Re-ranking**: The retrieved documents are re-ranked using a cross-encoder model to ensure the most relevant documents are selected.
5. **Answer Generation**: The system sends the top-ranked documents along with the user's query to Ollama, which generates a response based on the context.

## Acknowledgements

- **ChromaDB**: For vector storage and similarity search.
- **Ollama**: For context-based question answering.
- **Streamlit**: For building the web interface.
- **Sentence-Transformers**: For document re-ranking.
