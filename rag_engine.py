import os
import shutil
import json
import logging
from typing import List, Dict, Tuple, Optional
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from rich.console import Console

from utils import extract_pdf_content, create_chunks_with_metadata, format_source_reference

import asyncio

# Get logger
logger = logging.getLogger("rag_app.engine")

console = Console()

class RAGEngine:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the RAG engine.
        
        Args:
            api_key: Google API key
            model_name: Name of the Gemini model to use
        """

        # Ensure an event loop exists for Streamlit ScriptRunner thread
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        logger.info(f"Initializing RAG Engine with model: {model_name}")
        self.api_key = api_key
        self.model_name = model_name
        self.vector_store_path = "vector_store"
        
        logger.debug("Initializing Google Generative AI Embeddings")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", #"models/FinBert",
            google_api_key=api_key,
            task_type="retrieval_query"
        )
        
        logger.debug(f"Initializing ChatGoogleGenerativeAI with model: {model_name}")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.2,
            convert_system_message_to_human=True
        )
        
        # Create vector store directory if it doesn't exist
        os.makedirs(self.vector_store_path, exist_ok=True)
        logger.debug(f"Ensured vector store directory exists: {self.vector_store_path}")
        
        # Dictionary to store vector stores by PDF name
        self.vector_stores = {}
        self.load_existing_vector_stores()
    
    def load_existing_vector_stores(self):
        """
        Load existing vector stores from disk.
        """
        logger.info("Loading existing vector stores")
        if not os.path.exists(self.vector_store_path):
            logger.warning(f"Vector store path does not exist: {self.vector_store_path}")
            return
        
        items = os.listdir(self.vector_store_path)
        logger.debug(f"Found {len(items)} items in vector store directory")
            
        for item in items:
            item_path = os.path.join(self.vector_store_path, item)
            if os.path.isdir(item_path):
                logger.debug(f"Attempting to load vector store: {item}")
                try:
                    vector_store = FAISS.load_local(
                        item_path,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    self.vector_stores[item] = vector_store
                    logger.info(f"Successfully loaded vector store for {item}")
                    console.print(f"[green]Loaded vector store for {item}[/green]")
                except Exception as e:
                    error_msg = f"Error loading vector store {item}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    console.print(f"[red]{error_msg}[/red]")
    
    def process_pdf(self, pdf_path: str) -> bool:
        """
        Process a PDF file and create a vector store.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Boolean indicating success or failure
        """
        logger.info(f"Starting to process PDF: {pdf_path}")
        try:
            # Extract PDF name for vector store identification
            pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
            logger.debug(f"PDF name for vector store: {pdf_name}")
            
            # Extract content from PDF
            logger.debug(f"Extracting content from PDF: {pdf_path}")
            pages = extract_pdf_content(pdf_path)
            if not pages:
                logger.error(f"Failed to extract content from PDF: {pdf_path}")
                return False
            logger.debug(f"Successfully extracted {len(pages)} pages from PDF")
                
            # Create chunks with metadata
            logger.debug(f"Creating chunks with metadata from {len(pages)} pages")
            chunks = create_chunks_with_metadata(pages)
            if not chunks:
                logger.error("Failed to create chunks from PDF pages")
                return False
            logger.debug(f"Successfully created {len(chunks)} chunks")
                
            # Create vector store
            texts = [chunk["content"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            
            logger.debug(f"Creating FAISS vector store with {len(texts)} text chunks")
            vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            logger.debug("Successfully created FAISS vector store")
            
            # Save vector store to disk
            vector_store_dir = os.path.join(self.vector_store_path, pdf_name)
            logger.debug(f"Vector store directory: {vector_store_dir}")
            
            if os.path.exists(vector_store_dir):
                logger.debug(f"Removing existing vector store directory: {vector_store_dir}")
                shutil.rmtree(vector_store_dir)
                
            os.makedirs(vector_store_dir, exist_ok=True)
            logger.debug(f"Saving vector store to: {vector_store_dir}")
            vector_store.save_local(vector_store_dir)
            logger.debug("Vector store saved successfully")
            
            # Add to dictionary of vector stores
            self.vector_stores[pdf_name] = vector_store
            logger.debug(f"Added vector store to dictionary with key: {pdf_name}")
            
            success_msg = f"Successfully processed {pdf_name} and created vector store"
            logger.info(success_msg)
            console.print(f"[green]{success_msg}[/green]")
            return True
            
        except Exception as e:
            error_msg = f"Error processing PDF {pdf_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            console.print(f"[red]{error_msg}[/red]")
            return False
    
    def get_available_sources(self) -> List[str]:
        """
        Get list of available vector store sources.
        
        Returns:
            List of source names
        """
        return list(self.vector_stores.keys())
    
    def query(self, query: str, sources: List[str] = None, chat_history: List[Dict] = None) -> Dict[str, str]:
        """
        Query the RAG system.
        
        Args:
            query: User query
            sources: List of source names to query (if None, query all)
            chat_history: List of previous messages in the conversation
            
        Returns:
            Dictionary with answer and sources
        """
        logger.info(f"Processing query: '{query}'")
        logger.debug(f"Chat history length: {len(chat_history) if chat_history else 0}")
        
        if not self.vector_stores:
            logger.warning("No vector stores available for query")
            return {
                "answer": "No documents have been processed yet. Please upload and process a PDF first.",
                "sources": []
            }
            
        # If no sources specified, use all available sources
        if not sources:
            sources = list(self.vector_stores.keys())
            logger.debug(f"No sources specified, using all available: {sources}")
            
        # Filter to only include available sources
        sources = [s for s in sources if s in self.vector_stores]
        logger.debug(f"Using sources: {sources}")
        
        if not sources:
            logger.warning("None of the specified sources are available")
            return {
                "answer": "None of the specified sources are available. Please select valid sources.",
                "sources": []
            }
        
        # Initialize chat history if None
        if chat_history is None:
            chat_history = []
            
        # Combine results from multiple sources with improved retrieval
        all_docs = []
        source_info = []
        
        # Hybrid retrieval approach
        logger.info("Starting document retrieval process")
        for source in sources:
            vector_store = self.vector_stores[source]
            logger.debug(f"Querying vector store: {source}")

            # ✅ Use FAISS similarity search with scores
            logger.debug(f"Performing similarity search WITH SCORE for query: '{query}'")
            results = vector_store.similarity_search_with_score(query, k=15)
            logger.debug(f"Retrieved {len(results)} (doc, score) pairs from {source}")

            scored_docs = []
            for doc_idx, (doc, score) in enumerate(results):
                # store FAISS similarity score in metadata
                doc.metadata["score"] = str(round(score*100, 2)) + " %"
                scored_docs.append((doc, score))
                logger.debug(f"Doc {doc_idx} from {source} FAISS score: {score}")

            # Sort and take top 10
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            top_docs = [doc for doc, _ in scored_docs[:10]]

            all_docs.extend(top_docs)

            for doc in top_docs:
                # ✅ now includes score in reference
                source_ref = format_source_reference(doc.metadata, include_score=True)
                source_info.append(source_ref)
                logger.debug(f"Added source reference: {source_ref}")
        
        # Remove duplicates while preserving order
        unique_sources = []
        for source in source_info:
            if source not in unique_sources:
                unique_sources.append(source)
        source_info = unique_sources[:10]  # Limit to top 10
        logger.debug(f"Final source references: {source_info}")
        
        # Create context from documents with better formatting
        logger.debug("Creating context from retrieved documents")
        context_parts = []
        for i, doc in enumerate(all_docs[:10]):  # Limit to top 10 most relevant documents
            # Format each document with clear separation
            context_parts.append(f"Document {i+1}:\n{doc.page_content}\n")
            
            # Log document content (truncated for readability)
            content_sample = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            logger.debug(f"Document {i+1} content sample: {content_sample}")
            logger.debug(f"Document {i+1} metadata: {doc.metadata}")
            
            # Print full document content to console for debugging
            console.print(f"[bold blue]Retrieved Document {i+1}:[/bold blue]")
            console.print(f"[yellow]Page: {doc.metadata.get('page', 'Unknown')} | Source: {doc.metadata.get('source', 'Unknown')}[/yellow]")
            console.print(doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content)
            console.print("[bold green]---[/bold green]")
        
        context = "\n---\n".join(context_parts)
        logger.debug(f"Total context length: {len(context)} characters")
        
        # Convert chat history to LangChain message format
        formatted_history = []
        for message in chat_history:
            if message["role"] == "user":
                formatted_history.append(HumanMessage(content=message["content"]))
                logger.debug(f"Added user message to history: '{message['content']}'")
            elif message["role"] == "assistant":
                formatted_history.append(AIMessage(content=message["content"]))
                logger.debug(f"Added assistant message to history (length: {len(message['content'])} chars)")
        
        # Create improved RAG prompt
        system_prompt = """You are a helpful assistant that answers questions based on the provided context from the Company Annual Report. 
        
        Follow these rules:
        1. Only answer based on the context provided - do not use prior knowledge
        2. If the context doesn't contain the answer, say 'I don't have enough information to answer this question based on the provided documents'
        3. Provide detailed, accurate responses with specific information from the documents
        4. Do not make up or hallucinate any information
        5. Format your response in a clear, readable way with appropriate paragraphs and bullet points when needed
        6. Include relevant facts, figures, and statistics from the context when applicable
        7. Do not reference the context directly in your answer (e.g., don't say "According to Document 1...")
        8. Maintain a professional, informative tone appropriate for financial and business information
        9. If the question is vague, interpret it in the context of Company's business, operations, or financial performance
        10. Synthesize information from multiple documents when necessary to provide a complete answer
        11. When a user greets you (with words like "Hi", "Hello", "Hey"), respond with an appropriate greeting in return
        
        Remember that you are answering questions about Company's Annual Report, which contains financial information, business strategy, operational details, and corporate governance information.
        """
        
        human_prompt = """Question: {question}
        
        Context:
        {context}
        
        Answer the question based only on the provided context. Be thorough but concise.
        """
        
        logger.debug(f"System prompt: {system_prompt}")
        logger.debug(f"Human prompt template: {human_prompt}")
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content=human_prompt)
        ])
        
        # Create RAG chain with direct variable substitution
        # First, create a properly formatted human message with actual context and question
        formatted_human_prompt = human_prompt.format(question=query, context=context)
        
        # Create the chain with the formatted human prompt
        rag_chain = (
            {"context": lambda x: context, "question": lambda x: x["question"], "chat_history": lambda x: x["chat_history"]}
            | ChatPromptTemplate.from_messages([
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessage(content=formatted_human_prompt)
            ])
            | self.llm
            | StrOutputParser()
        )
        
        # Execute chain
        logger.info("Executing RAG chain with LLM")
        try:
            # Create the new prompt template with the already formatted human message
            new_prompt_template = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessage(content=formatted_human_prompt)
            ])
            
            # Log the actual formatted prompt that will be sent to the LLM
            formatted_prompt = new_prompt_template.format_messages(
                chat_history=formatted_history
            )
            
            logger.debug("Formatted prompt messages:")
            for idx, msg in enumerate(formatted_prompt):
                msg_type = type(msg).__name__
                content_sample = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
                logger.debug(f"Message {idx} ({msg_type}): {content_sample}")
                
                # Print the full formatted prompt to console for debugging
                console.print(f"\n[bold magenta]Prompt Message {idx} ({msg_type}):[/bold magenta]")
                console.print(msg.content)
                console.print("[bold green]===END OF MESSAGE===[/bold green]\n")
            
            answer = rag_chain.invoke({"question": query, "chat_history": formatted_history})
            logger.debug(f"LLM response (length: {len(answer)} chars): {answer[:200]}...")
            logger.info("Successfully generated answer from LLM")
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            logger.error(error_msg, exc_info=True)
            answer = f"An error occurred while generating the answer: {str(e)}"
        
        return {
            "answer": answer,
            "sources": source_info  # Return the source references for display in the UI
        }