import os
import streamlit as st
import tempfile
import logging
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional
from rich.console import Console


import asyncio
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


from rag_engine import RAGEngine
from utils import get_pdf_files

# Get logger
logger = logging.getLogger("rag_app.streamlit")

# Load environment variables
load_dotenv()

logger.info("Starting Streamlit RAG application")

# Set page configuration
st.set_page_config(
    page_title="RAG AI App",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("📈 Cognizant Annual Report Assistant")
st.markdown("""
    This application allows you to upload PDF documents, process them, and ask questions about their content.
    The system uses Retrieval-Augmented Generation (RAG) to provide accurate answers with source references.
""")

# Sidebar for configuration and document management
with st.sidebar:
    st.header("Configuration")
    logger.debug("Rendering configuration sidebar")

    # API Key input
    api_key = os.getenv("GOOGLE_API_KEY", "")
    logger.debug(f"API key from environment: {'Present (masked)' if api_key else 'Not found'}")

    input_api_key = st.text_input("Google API Key", value=api_key, type="password",
                                  help="Enter your Google Gemini API key. If left empty, the key from .env file will be used.")

    # Use input API key if provided, otherwise use the one from .env
    if input_api_key.strip():
        api_key = input_api_key
        logger.debug("Using API key from user input")
    elif not api_key:
        logger.error("No API key found in environment or user input")
        st.error(
            "Google API key not found. Please enter your API key above or set the GOOGLE_API_KEY environment variable in the .env file.")
        st.stop()

    # Model selection
    model_options = [
        "gemini-2.0-flash",
        "gemini-2.0-lite",
        "gemini-1.5-flash",
        "gemini-1.5-pro"
    ]
    selected_model = st.selectbox("Select Gemini Model", model_options, index=0,
                                  help="Choose which Gemini model to use for generating responses.")
    logger.debug(f"Selected model: {selected_model}")

    # Initialize RAG engine with selected configuration
    if st.button("Apply Configuration"):
        logger.info(f"Applying new configuration with model: {selected_model}")
        with st.spinner("Initializing RAG engine with new configuration..."):
            try:
                st.session_state.rag_engine = RAGEngine(api_key=api_key, model_name=selected_model)
                success_msg = f"Successfully configured RAG engine with model: {selected_model}"
                logger.info(success_msg)
                st.success(success_msg)
            except Exception as e:
                error_msg = f"Failed to initialize RAG engine: {str(e)}"
                logger.error(error_msg, exc_info=True)
                st.error(error_msg)

    # Initialize RAG engine if not already in session state
    if "rag_engine" not in st.session_state:
        logger.info("Initializing RAG engine for the first time")
        try:
            st.session_state.rag_engine = RAGEngine(api_key=api_key, model_name=selected_model)
            logger.info("Successfully initialized RAG engine")
        except Exception as e:
            error_msg = f"Failed to initialize RAG engine: {str(e)}"
            logger.error(error_msg, exc_info=True)
            st.error(error_msg)
            st.stop()

    rag_engine = st.session_state.rag_engine
    logger.debug(f"Using RAG engine with model: {rag_engine.model_name}")

    st.header("Document Management")

    # Sidebar for PDF upload and processing (continued)

    # Upload new PDF
    st.subheader("Upload New Document")
    logger.debug("Rendering document upload section")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        logger.info(f"PDF uploaded: {uploaded_file.name}, size: {uploaded_file.size} bytes")
        if st.button("Process Document"):
            logger.info(f"Starting to process document: {uploaded_file.name}")
            with st.spinner("Processing document..."):
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                    logger.debug(f"Created temporary file at: {tmp_file_path}")

                # Copy to data directory
                os.makedirs("data", exist_ok=True)
                target_path = os.path.join("data", uploaded_file.name)
                logger.debug(f"Copying file to: {target_path}")
                with open(target_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Process PDF
                logger.info(f"Processing PDF: {target_path}")
                success = rag_engine.process_pdf(target_path)

                # Clean up temporary file
                logger.debug(f"Cleaning up temporary file: {tmp_file_path}")
                os.unlink(tmp_file_path)

                if success:
                    success_msg = f"Successfully processed {uploaded_file.name}"
                    logger.info(success_msg)
                    st.success(success_msg)
                else:
                    error_msg = f"Failed to process {uploaded_file.name}"
                    logger.error(error_msg)
                    st.error(error_msg)

    # Process existing PDFs
    st.subheader("Process Existing Documents")
    pdf_files = get_pdf_files("data")
    logger.debug(f"Found {len(pdf_files)} PDF files in data directory")

    if pdf_files:
        pdf_options = [os.path.basename(pdf) for pdf in pdf_files]
        selected_pdf = st.selectbox("Select PDF to process", pdf_options)
        logger.debug(f"User selected PDF: {selected_pdf}")

        if st.button("Process Selected Document"):
            logger.info(f"Starting to process selected document: {selected_pdf}")
            with st.spinner("Processing document..."):
                selected_pdf_path = os.path.join("data", selected_pdf)
                logger.debug(f"Processing PDF at path: {selected_pdf_path}")
                success = rag_engine.process_pdf(selected_pdf_path)

                if success:
                    success_msg = f"Successfully processed {selected_pdf}"
                    logger.info(success_msg)
                    st.success(success_msg)
                else:
                    error_msg = f"Failed to process {selected_pdf}"
                    logger.error(error_msg)
                    st.error(error_msg)
    else:
        logger.debug("No PDF files found in data directory")
        st.info("No PDF files found in the data directory.")

# Main content area
# Display available sources and query interface
logger.debug("Getting available sources from RAG engine")
available_sources = rag_engine.get_available_sources()
logger.debug(f"Found {len(available_sources)} available sources")

# Initialize chat history in session state if not already present
if "chat_history" not in st.session_state:
    logger.debug("Initializing chat history")
    st.session_state.chat_history = []
else:
    logger.debug(f"Chat history exists with {len(st.session_state.chat_history)} messages")

if available_sources:
    st.header("Ask Questions")

    logger.debug("Rendering question interface")

    # Select sources to query
    st.subheader("Select Sources")
    selected_sources = st.multiselect(
        "Choose which documents to query",
        available_sources,
        default=available_sources
    )
    logger.debug(f"User selected {len(selected_sources)} sources for querying")

    # Add a button to clear chat history
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("Clear Chat History"):
            logger.info("Clearing chat history")
            st.session_state.chat_history = []
            st.rerun()

    # Display chat history
    logger.debug(f"Displaying {len(st.session_state.chat_history)} chat messages")
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            logger.debug(f"Rendering message {i + 1}: role={message['role']}, content_length={len(message['content'])}")
            if message["role"] == "user":
                st.markdown(
                    f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>You:</strong> {message['content']}</div>",
                    unsafe_allow_html=True)
            else:  # assistant
                st.markdown(
                    f"<div style='background-color: #e6f7ff; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>Assistant:</strong> {message['content']}</div>",
                    unsafe_allow_html=True)

                # Display sources if available
                if "sources" in message and message["sources"]:
                    with st.expander("Sources"):
                        for source in message["sources"]:
                            st.markdown(f"- {source}")
                    logger.debug(f"Displayed {len(message['sources'])} sources for message {i + 1}")

    # Query input

    # Suggested Questions Section
    st.subheader("💡 Suggested Questions")
    # Dictionary of categories
    question_categories = {
        "Finance & PNL": [
            "What was Cognizant’s total revenue for the last fiscal year?",
            "How did revenue change compared to the previous year?",
            "What was the net income for the year?",
            "Can you break down the revenue by business segment?",
            "What was the year-over-year growth in operating income?",
            "How much did Cognizant spend on research and development?"
        ],
        "Company Overview": [
            "Who are Cognizant’s key executives and leadership team?",
            "What is Cognizant’s primary business model?",
            "Which regions contributed most to the company’s revenue?",
            "What are the main industries Cognizant serves?"
        ],
        "Operational & Strategic Insights": [
            "What were the key highlights of Cognizant’s annual performance?",
            "What major acquisitions or partnerships occurred during the year?",
            "What are the company’s stated strategic priorities for next year?"
        ],
        "Sustainability & ESG": [
            "What are Cognizant’s ESG initiatives?",
            # "How much did the company invest in sustainability programs?",
            "What diversity and inclusion metrics are reported?"
        ]
    }
    # Loop through each category
    for category, questions in question_categories.items():
        st.markdown(f"**{category}**")  # Category title in bold
        cols = st.columns(3)
        for idx, question in enumerate(questions):
            if cols[idx % 3].button(question, key=f"{category}_{idx}"):
                query = question
                st.session_state.chat_history.append({"role": "user", "content": query})
                with st.spinner("Generating answer..."):
                    result = rag_engine.query(
                        query=query,
                        sources=selected_sources,
                        chat_history=st.session_state.chat_history[:-1]
                    )
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })
                st.rerun()

    st.subheader("Your Question")
    query = st.text_input("Enter your question about the document(s)")

    if query and st.button("Submit"):
        logger.info(f"User submitted query: '{query}'")
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        logger.debug("Added user message to chat history")

        with st.spinner("Generating answer..."):
            logger.info(f"Querying RAG engine with: '{query}'")
            logger.debug(f"Using {len(selected_sources)} selected sources")
            logger.debug(f"Passing chat history with {len(st.session_state.chat_history) - 1} messages")
            # Pass the chat history to the query function
            result = rag_engine.query(
                query=query,
                sources=selected_sources,
                chat_history=st.session_state.chat_history[:-1]  # Exclude the current query
            )
            logger.debug(f"Received response from RAG engine, answer length: {len(result['answer'])}")
            logger.debug(f"Received {len(result['sources'])} sources from RAG engine")

            # Add assistant response to chat history with sources
            logger.debug("Adding assistant message to chat history with sources")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"]
            })
            logger.debug(f"Chat history now has {len(st.session_state.chat_history)} messages")

        # Rerun to update the UI with the new messages
        logger.debug("Rerunning Streamlit to update UI with new messages")
        st.rerun()
else:
    logger.debug("No sources available, showing instructions")
    st.info("No documents have been processed yet. Please upload and process a PDF first.")

# Footer
st.markdown("---")
st.markdown(f"""
    <div style="text-align: center">
        <p>Built By Shraddha Nimbwale</p>
    </div>
""", unsafe_allow_html=True)
logger.debug("Rendered footer with model information")