"""
ARIA Vector Store Test Suite
Tests for the vector_store.py module
"""
import os
import sys
import shutil
import logging
from typing import List

# add src to Python path
sys.path.insert(0, 'src')

from langchain_core.documents import Document

# Assuming vector_store.py is in the same directory
from embeddings.vector_store import ARIAVectorStore, VECTOR_STORE_PATH

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

def create_sample_documents() -> List[Document]:
    """Generates a list of sample LangChain Document objects for testing."""
    logger.info("Creating sample documents...")
    docs = [
        Document(page_content="The project plan outlines four main phases."),
        Document(page_content="Phase 1 focuses on OCR and text processing, including chunking."),
        Document(page_content="The tech stack includes PaddleOCR and FAISS."),
        Document(page_content="Ollama will be used for answer generation."),
        Document(page_content="The deployment will use Docker and Render's free tier.")
    ]
    logger.info(f"✅ Created {len(docs)} sample documents.")
    return docs

def test_vector_store_initialization():
    """Test the initialization of the ARIAVectorStore class."""
    logger.info("🔍 Testing vector store initialization...")
    store = ARIAVectorStore()
    if store.embeddings and store.vector_store:
        logger.info("✅ Vector store and embeddings initialized successfully.")
        return store
    else:
        raise AssertionError("Vector store or embeddings failed to initialize.")
      
def test_add_documents(store: ARIAVectorStore, docs: List[Document]):
    """Test adding documents to the vector store."""
    logger.info("📄 Testing document addition...")
    
    # get the count of before and after adding documents
    initial_count = len(store.vector_store.docstore._dict)
    store.add_documents(docs)
    final_count = len(store.vector_store.docstore._dict)
    # compare
    assert final_count == initial_count + len(docs), f"Expected {initial_count + len(docs)}, got {final_count} instead"
    
    # verify the index file was created
    assert os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss"))
    logger.info("✅ Documents added successfully and index file created.")
    
def test_similarity_search(store: ARIAVectorStore):
    logger.info("🔍 Testing similarity search...")
    query = "What tech stack is used in the project?"
    results = store.similarity_search(query, k=3)
    
    assert len(results) > 0, "❌ No relevant document found"
     
    logger.info("✅ Similarity search successful and relevant results found.")
    logger.info(f"Search results for '{query}': ")
    for i, doc in enumerate(results):
        print(f"    Result {i} (source: {doc.metadata.get('source', 'N/A')}): {doc.page_content[:150]}...")

if __name__ == "__main__":
    try:
        # Step 1: Set up a clean test environment
        setup_test_env()
        
        # Step 2: Create sample data
        sample_documents = create_sample_documents()
        
        # Step 3: Test initialization
        vector_store_instance = test_vector_store_initialization()
        
        # Step 4: Test adding documents
        test_add_documents(vector_store_instance, sample_documents)
        
        # Step 5: Test similarity search
        test_similarity_search(vector_store_instance)

        print("\n🎉 All tests for vector_store.py passed successfully!")
        print("🎉 The ARIA RAG system is ready for retrieval and search operations. Let's go!")
        
    except AssertionError as e:
        logger.error(f"❌ Assertion Failed: {e}")
        logger.error("Tests failed. Please check the vector_store.py implementation.")
        
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred: {e}")
        logger.error("Please check your environment and dependencies.")
    
    