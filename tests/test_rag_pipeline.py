"""
ARIA Vector Store Test Suite
Tests for the vector_store.py module
"""
import os
import sys
import shutil
import logging

# add src to Python path
sys.path.insert(0, 'src')

# import modules
from embeddings.vector_store import VECTOR_STORE_PATH, INDEX_NAME, ARIAVectorStore
from retrieval.rag_pipeline import ARIARAGPipeline

# Configure logging to show all messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_test_env():
    """Sets up a clean test environment by removing any existing vector store."""
    logger.info("🏗️ Setting up test environment...")

    if os.path.exists(VECTOR_STORE_PATH):
        try:
            shutil.rmtree(VECTOR_STORE_PATH)
            logger.info(f"Existing vector store directory '{VECTOR_STORE_PATH}' removed.")
        except OSError as e:
            logger.error(f"Error removing directory: {e}")
            raise
    logger.info("✅ Test environment ready.")
    
def setup_pinecone_test_env():
    """Sets up a clean test environment by removing any existing vector store."""
    logger.info("🏗️ Setting up test environment...")
    store = ARIAVectorStore()
    store.pinecone_client.delete_index(INDEX_NAME)
    
    logger.info("✅ Test environment ready.")

def test_rag_pipeline():
    logger.info("🚀 Initializing ARIA RAG Pipeline...")
    pipeline = ARIARAGPipeline()
    try:
        # 1. Ingest a document
        document_to_ingest = "docs/tuberculosis_receipt.png"
        if os.path.exists(document_to_ingest):
            pipeline.ingest_document(document_to_ingest)
            print()
        else:
            logger.error(f"File not found: {document_to_ingest}. Please place it in the same directory.")
            return False

        print("-" * 50)
        
        #2. Ask questions
        if os.path.exists(document_to_ingest):
            questions = [
                "What university did the candidate graduate from?",
                "What is the candidate's main tech stack?",
                "How many years of experience does the candidate have?"
            ]
            
            for q in questions:
                print(f"**Question**: {q}")
                answer = pipeline.answer_query(q)   
                
                full_answer = ""
                for chunk in answer:
                    full_answer += chunk
                    # print the chunk to see the streaming effect
                    print(chunk, end="", flush=True)
                
                # print(f"**Answer**: {full_answer}")
                print()
                print("-" * 50)
                
        return True
    except Exception as e:
        logger.error(f"❌ Failed to ingest and answer question: {e}")
        raise
        
if __name__ == "__main__":
    print("🅰️ ARIA RAG Pipeline")
    print("Testing professional RAG Pipeline implementation")
    print("=" * 60)
    
    # set up test environment
    setup_pinecone_test_env()
    
    # run main test
    success = test_rag_pipeline()
    
    # print out message if success
    if success:
        print(f"\n{'='*60}")
        print("🎉 RAG Pipeline Testing COMPLETED!")
        
    # print out fail messages and fixes
    else:
        print("\n❌ Some tests failed. Check error messages above.")
        print("Common fixes:")
        print("1. Make sure virtual environment is activated")
        print("2. Check if all packages are installed: pip list")
        print("3. Try running: pip install -r requirements.txt --force-reinstall")