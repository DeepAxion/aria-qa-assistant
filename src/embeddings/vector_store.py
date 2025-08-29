"""
ARIA Vector Store Manager 
Handles all vector store operations with FAISS and sentence-transformers
"""
import os
import logging
import faiss
from typing import List
from pathlib import Path
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Choose a local embedding model that is free and has a permissive license.
# 'all-MiniLM-L6-v2' is a good, small, and fast choice for a start. 
# load environment variables from .env file
load_dotenv()

EMBEDDING_MODEL_NAME = "text-embedding-3-small"
VECTOR_STORE_PATH = "data/faiss_index"
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

class ARIAVectorStore:
    """
    Manages the FAISS vector store for ARIA RAG system.
    """
    def __init__(self):
        
        logger.info("🅰️ Initializing ARIA Vector Store...")
        
        # initialize embedding model
        self.embeddings = self._get_embedding_model()
        
        # initialize pinecone client
        try:
            self.pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
            
            # check if index exists
            if INDEX_NAME in self.pinecone_client.list_indexes().names():
                logger.info(f"✅ Pinecone index '{INDEX_NAME}' already exists. Skipping creation.")
            else: 
                self._create_pinecone_index()
                    
            logger.info("✅ Pinecone connection successful")
        except Exception as e:
            logger.error(f"😟 Failed to connect to Pinecone: {e}")
        
        # load or create the FAISS Index
        # self.vector_store = self._load_or_create_vector_store()
        self.vector_store = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME, embedding=self.embeddings, namespace=""
        )
        
        logger.info("✅ Vector Store initialized successfully! ")

    def _get_embedding_model(self):
        """Initializes and returns the local embedding model."""
        # using openai embedding model and chat model for ease and speed
        try:
            model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
            
            logger.info(f"🛞 Loading Embedding Model...")
            return model
        
        except Exception as e:
            logger.error(f"😟 Failed to load Embedding Model: {e}")
        raise
        
        ### we can also use local embedding model 
        # try:
        #     # use HuggingfaceEmbeddings to download and load the model
        #     model = HuggingFaceEmbeddings(
        #         model_name="sentence-transformers/all-MiniLM-L6-v2",
        #         model_kwargs={"device": "cpu"},
        #         encode_kwargs={"normalize_embeddings": True}
        #     )
            
        #     logger.info(f"🛞 Loading Embedding Model: {EMBEDDING_MODEL_NAME}")
        #     return model
        # except Exception as e:
        #     logger.error(f"Failed to load Embedding Model: {e}")
        #     raise

    def _create_pinecone_index(self):
        """Creates a Pinecone index if it doesn't exist."""
        logger.info(f"🆕 Creating new Pinecone index: {INDEX_NAME}")
        self.pinecone_client.create_index(
            name=INDEX_NAME,
            dimension=len(self.embeddings.embed_query("hello world")),
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        logger.info("✅ Index created successfully")
        
    def _load_or_create_vector_store(self):
        """Loads the FAISS index from disk or creates a new one."""
        
        # define the paths for the FAISS index and docstore files
        index_path = Path(VECTOR_STORE_PATH) / "index.faiss"
        docstore_path = Path(VECTOR_STORE_PATH) / "index.pkl"
        
        # create the folders if not exists
        Path(VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)#create parent folder if it doesn't exist
        
        try:
            # check if both the index and docstore files exist
            if index_path.exists() and docstore_path.exists():
                logger.info(f"Loading existing FAISS index from {VECTOR_STORE_PATH}")
                return FAISS.load_local(VECTOR_STORE_PATH, self.embeddings, allow_dangerous_deserialization=True)
            else:
                 # create a new empty FAISS index in memory
                logger.warning("🆕 No existing FAISS index found. Creating a new empty index...")
                # call embed_query on any text to get the dimension as faiss.IndexFlatL2 requires dim as an argument
                dim = len(self.embeddings.embed_query("Hello world"))
                # index to use, faiss needs to know the size of each vector to allocate memory and calculate the distance
                index = faiss.IndexFlatL2(dim)
                
                # create vector store
                vector_store = FAISS(
                    embedding_function=self.embeddings, # embedding model
                    index=index, 
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={}
                )
                
                return vector_store

        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise

    def add_documents(self, documents: List[Document]):
        """Add a list of documents to vector store"""
        try:
            if not isinstance(documents, list):
                documents = [documents]

            # add document to store, passing the unique user's namespace from user ID
            self.vector_store.add_documents(documents)
            # save the added document to vector store
            # self.vector_store.save_local(VECTOR_STORE_PATH)
            # logger.info(f"Added {len(documents)} documents to FAISS vector store and saved to disk.")
            logger.info(f"Added {len(documents)} documents to Pinecone index.")

        except Exception as e:
            logger.error(f"Failed to add documents to Pinecone: {e}")
            raise
        
    def similarity_search(self, query: str, k: int = 4, ) -> List[Document]:
        """Performs a similarity search on the vector store."""
        logger.info(f"🔍 Searching for {query} ... ")
        
        # search logic is handled by the FAISS index and the embedding model
        found_docs = self.vector_store.similarity_search(query, k)
        
        if found_docs:
            logger.info(f"📃 Found {len(found_docs)} relevant documents!")
        else:
            logger.warning("😟 No relevant documents found")
            
        return found_docs

    def clear_store(self):
        """clear the vector store"""
        try:
            import shutil
            shutil.rmtree(VECTOR_STORE_PATH)
            self.vector_store = FAISS.from_documents(documents=[], embedding=self.embeddings)
            logger.info("🗑️ Vector store cleared successfully.")
        except FileNotFoundError:
            logger.warning("Vector store directory not found, nothing to clear.")
        except Exception as e:
            logger.error(f"Failed to clear vector store: {e}")
            raise
        
    def clean_pinecone_store(self):
        """Deletes all vectors from the Pinecone index."""
        try:
            index = self.pinecone_client.Index(INDEX_NAME)
            index.delete(delete_all=True)
            logger.info("🗑️ Pinecone index cleared successfully.")
        except Exception as e:
            logger.error(f"Failed to clear Pinecone index: {e}")
            raise
        
    
        
        