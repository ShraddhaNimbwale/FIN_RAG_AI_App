import os
import PyPDF2
import logging
from typing import List, Dict, Tuple, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_app")

console = Console()

def extract_pdf_content(pdf_path: str) -> List[Dict[str, str]]:
    """
    Extract text content from a PDF file with page numbers and headers.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of dictionaries with page content, page number, and headers
    """
    try:
        logger.info(f"Starting PDF extraction for: {pdf_path}")
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        pdf_name = os.path.basename(pdf_path)
        pages = []
        
        logger.debug(f"PDF has {len(pdf_reader.pages)} pages")
        
        for i, page in enumerate(pdf_reader.pages):
            logger.debug(f"Processing page {i+1}")
            text = page.extract_text()
            if text.strip():
                # Try to extract header (more intelligently)
                lines = text.split('\n')
                logger.debug(f"Page {i+1} has {len(lines)} lines")
                
                # Look for potential headers in the first few lines
                potential_headers = []
                for j in range(min(3, len(lines))):
                    line = lines[j].strip()
                    # Headers are typically short and may contain specific patterns
                    if line and len(line) < 100 and not line.endswith('.') and not line.startswith('  '):
                        potential_headers.append(line)
                        logger.debug(f"Potential header found: {line}")
                
                # Use the most likely header
                header = potential_headers[0] if potential_headers else ""
                logger.debug(f"Selected header: {header}")
                
                # Clean up the text by removing excessive whitespace
                cleaned_text = '\n'.join([line.strip() for line in lines if line.strip()])
                
                # Log a sample of the text (first 200 chars)
                text_sample = cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text
                logger.debug(f"Page {i+1} text sample: {text_sample}")
                
                pages.append({
                    "content": cleaned_text,
                    "page_number": i + 1,
                    "header": header,
                    "source": pdf_name
                })
                
        console.print(f"[green]Successfully extracted {len(pages)} pages from {pdf_name}[/green]")
        logger.info(f"Successfully extracted {len(pages)} pages from {pdf_name}")
        return pages
    except Exception as e:
        error_msg = f"Error extracting content from {pdf_path}: {str(e)}"
        console.print(f"[red]{error_msg}[/red]")
        logger.error(error_msg, exc_info=True)
        return []

def create_chunks_with_metadata(pages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Create chunks from PDF pages with simplified metadata.
    
    Args:
        pages: List of dictionaries with page content and metadata
        
    Returns:
        List of dictionaries with chunks and basic metadata
    """
    logger.info(f"Starting chunking process for {len(pages)} pages")
    chunks = []
    
    # Configure the text splitter with smaller chunks for better accuracy
    chunk_size = 30000  # 30,000 characters per chunk
    chunk_overlap = 600  # 600 characters overlap
    logger.debug(f"Configuring text splitter with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    # Define separators for the text splitter
    separators = ["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    logger.debug(f"Text splitter separators: {separators}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators
    )
    
    for page_idx, page in enumerate(pages):
        logger.debug(f"Processing page {page['page_number']} for chunking")
        
        # Create simplified metadata with only essential information
        metadata = {
            "page": page["page_number"],
            "source": page["source"]
        }
        logger.debug(f"Page metadata: {metadata}")
        
        # Create chunks with the simplified metadata
        logger.debug(f"Creating chunks for page {page['page_number']} with content length {len(page['content'])}")
        page_chunks = text_splitter.create_documents(
            texts=[page["content"]],
            metadatas=[metadata]
        )
        logger.debug(f"Created {len(page_chunks)} chunks for page {page['page_number']}")
        
        for chunk_idx, chunk in enumerate(page_chunks):
            # Log chunk details
            chunk_sample = chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content
            logger.debug(f"Chunk {chunk_idx+1} from page {page['page_number']}, length: {len(chunk.page_content)} chars")
            logger.debug(f"Chunk sample: {chunk_sample}")
            
            chunks.append({
                "content": chunk.page_content,
                "metadata": chunk.metadata
            })
    
    console.print(f"[green]Created {len(chunks)} chunks from {len(pages)} pages[/green]")
    logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks

def get_pdf_files(directory: str) -> List[str]:
    """
    Get all PDF files in a directory.
    
    Args:
        directory: Directory path
        
    Returns:
        List of PDF file paths
    """
    pdf_files = []
    for file in os.listdir(directory):
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(directory, file))
    return pdf_files

def format_source_reference(metadata: Dict[str, any], include_score: bool = False) -> str:
    """
    Format source reference from metadata.
    """
    source = metadata.get("source", "Unknown source")
    page = metadata.get("page", "Unknown page")
    reference = f"Source: {source}, Page: {page}"
    if include_score and "score" in metadata:
        reference += f" (Similarity Score: {metadata['score']})"
    return reference