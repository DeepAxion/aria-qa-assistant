"""
ARIA RAG Pipeline
Implements the core RAG workflow with local models.
"""
import logging
import os
import sys
from pathlib import Path
from typing import List, TypedDict

# from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import our custom components
from ocr.processor import ARIAOCRProcessor
from embeddings.vector_store import ARIAVectorStore

# add src to Python path
sys.path.insert(0, 'src')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ARIAState(TypedDict):
    """
    Represents the state of the RAG pipeline.
    """
    question: str
    documents: List[Document]
    
class ARIARAGPipeline:
    """Main class for rag pipeline, from document ingestion to answer generation"""
    def __init__(self):
        logger.info("🚀 Initializing ARIA RAG Pipeline...")
        
        # initialize our custom components
        self.ocr_processor = ARIAOCRProcessor()
        self.vector_store_manager = ARIAVectorStore()
        
        # initialize the local LLM
        self.llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
        
        # initialize text splitter (chunker)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False
        )
        
        # define retrieval and generation chain
        self.rag_chain = self._build_rag_chain()
        
        logger.info("✅ RAG Pipeline initialized")
        
    def _build_rag_chain(self):
        """Build the LangChain RAG pipeline."""
        
        # define prompt template 
        prompt_template = ChatPromptTemplate.from_template(
            """
            You are a helpful and informative assistant. Answer the user's question based ONLY on the following context, providing citations(the page number and the document name) for each piece of information you use.
            Keep answers concise and relevant.
            
            If the answer cannot be found in the context, just say "I don't know." DO NOT make up any information.
            
            Context:
            {context}
            
            Question: {question}
            
            Example Citations: (Page number, Document)
            
            """
        )
        
        # the retriever component uses our custom vector manager 
        retriever = self.vector_store_manager.vector_store.as_retriever()
        
        # The LangChain expression language chain
        # 1. Take question, pass to retriever to get documents
        # 2. Combine the documents into a single string
        # 3. Pass the combined docs and question to the LLM with the prompt
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain

    def ingest_document(self, file_path: str):
        """
        Processes a document and adds its chunks to the vector store.
        
        Args:
            file_path: Path to the document to ingest.
        """
        
        logger.info(f"📥 Ingesting document: {file_path}")
        
        # Step 1: Process the file with OCR to get text as Langchain document
        try:
            ocr_docs = self.ocr_processor.get_langchain_documents(file_path)
            if not ocr_docs:
                logger.info("😟 No document was ingested")
                return
            
        except Exception as e:
            logger.error(f"Document ingestion failed during OCR process: {e}")
            raise
        
        # Step 2: Chunk the OCR document
        logger.info(f"🔪 Splitting {len(ocr_docs)} document(s) into chunks...")
        chunks = self.text_splitter.split_documents(ocr_docs)
        logger.info(f"Generated {len(chunks)} chunks")
        
        # Step 3: Add chunks to the vector store
        self.vector_store_manager.add_documents(chunks)
        logger.info("✅ Document ingestion completed.")
    
    def answer_query(self, question: str):
        """
        Answers a question using the RAG pipeline.
        
        Args:
            question: The user's question.
            
        Returns:
            The generated answer from the LLM.
        """
        
        logger.info(f"🗨️ Answering query: '{question}")
        try:
            # use streaming
            for chunk in self.rag_chain.stream(question):
                yield chunk
            
        except Exception as e:
            logger.error(f"❌ Failed to answer query: {e}")
            raise 
   
# if __name__ == "__main__":     
    
#     logger.info("🚀 Initializing ARIA RAG Pipeline...")
#     pipeline = ARIARAGPipeline()
#     try:
#         # 1. Ingest a document
#         document_to_ingest = "docs/test_resume.pdf"
#         if os.path.exists(document_to_ingest):
#             pipeline.ingest_document(document_to_ingest)
#         else:
#             logger.error(f"File not found: {document_to_ingest}. Please place it in the same directory.")

#         print("-" * 50)
        
#         #2. Ask questions
#         if os.path.exists(document_to_ingest):
#             questions = [
#                 "Where did the candidate graduate from?",
#                 "What is the candidate's main tech stack?",
#                 "How many years of experience does the candidate have?"
#             ]
            
#             for q in questions:
#                 print(f"**Question**: {q}")
#                 answer = pipeline.answer_query(q)
#                 full_answer = ""
#                 for chunk in answer:
#                     full_answer += chunk
#                     # print the chunk to see the streaming effect
#                     print(chunk, end="", flush=True)
                
#                 print(f"**Answer**: {full_answer}")
#                 print("-" * 50)
#     except Exception as e:
#         logger.error(f"❌ Failed to ingest and answer question: {e}")
#         raise
        