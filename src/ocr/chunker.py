"""
Splits document text into manageable chunks with metadata.
"""

import logging
from typing import List, Dict, Optional, Any
import re

logger = logging.getLogger(__name__)

class ARIAChunker:
    """
    Chunks large documents into smaller, overlapping chunks
    for processing in a RAG pipeline.
    """
    
    def __init__(self, chunk_size: int=1000, chunk_overlap: int=100):
        """
        Initializes the document chunker.

        Args:
            chunk_size: The target size of each text chunk (in characters).
            chunk_overlap: The number of characters to overlap between chunks.
        """
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"✅ Initialize ARIAChunker with chunk_size={chunk_size} and chunk_overlap={chunk_overlap}")
        
    def chunk_text(self, text: str, metadata: Dict[str, any]) -> List[Dict[str, Any]]:
        """
        Splits a given text into overlapping chunks and attaches metadata.

        Args:
            text: The full text of the document.
            metadata: A dictionary of metadata to attach to each chunk.

        Returns:
            A list of dictionaries, where each dictionary represents a chunk
            and its associated metadata.
        """
        
        if not text:
            return []
        
        chunks = []
        text_length = len(text)
        start = 0
        
        try:
            while start < text_length:
                # get the end of chunk
                end = start + self.chunk_size
                # get the chunk 
                chunk_text = text[start:end]
                
                # find the page of the chunk
                page_number = self._get_page_number_for_chunk(chunk_text)
                
                # create a unique ID for the chunk
                chunk_id = f"{metadata.get('file_name', 'unknown')}_chunk_{len(chunks)}"
                
                chunk = {
                    'id': chunk_id,
                    'text': chunk_text,
                    'source': metadata.get('file_name', 'unknown'),
                    'page_number': page_number,
                    'metadata': metadata
                }
                chunks.append(chunk)
                
                # move the pointer
                start += self.chunk_size - self.chunk_overlap
                
            return chunks
        
        except Exception as e:
            logger.error(f"❌ Unable to chunk document: {e}")
            raise
            
    def _get_page_number_for_chunk(self, chunk_text: str) -> Optional[int]:
        """
        Extracts a page number from the chunk's text based on the
        "--- Page X ---" pattern.
        """
        # define pattern
        # it could be -page1- or ----page 1----
        pattern = r'-*\s*page\s+(\d+)\s*-*'
        match = re.search(pattern, chunk_text, re.IGNORECASE) # ignore case for the word 'page'
        if match:
            return int(match.group(1)) # we want to capture what's inside () which is a number in pattern (\d+)
        
        return None

