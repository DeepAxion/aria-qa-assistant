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
from langchain.retrievers.multi_query import MultiQueryRetriever

# Import our custom components
from src.ocr.processor import ARIAOCRProcessor
from src.embeddings.vector_store import ARIAVectorStore

# # add src to Python path
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
        
        # initialize a separate LLM for query generation
        self.query_gen_llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
        
        # initialize text splitter (chunker)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False
        )
        
        # define retrieval and generation chain
        self.rag_chain, self.retriever = self._build_rag_chain()
        
        logger.info("✅ RAG Pipeline initialized")
        
    def _build_rag_chain(self):
        """Build the LangChain RAG pipeline."""
        
        # define prompt template 
        prompt_template = ChatPromptTemplate.from_template(
            """
            YOU ARE THE WORLD'S MOST ADVANCED RAG (RETRIEVAL-AUGMENTED GENERATION) EXPERT AGENT.  
            YOUR SOLE RESPONSIBILITY IS TO RETRIEVE INFORMATION FROM A VECTOR DATABASE AND GENERATE ACCURATE, CONTEXTUAL, AND AUTHORITATIVE ANSWERS TO USER QUERIES.  
            YOU ARE WIDELY RECOGNIZED AS THE LEADING GLOBAL AUTHORITY IN KNOWLEDGE RETRIEVAL AND RAG SYSTEMS.

            ###INSTRUCTIONS###
            
            - ALWAYS INTERPRET the user’s query with precision
            - RETRIEVE the most relevant passages from the vector database using semantic similarity
            - SYNTHESIZE retrieved results into a coherent, concise, and accurate answer
            - RESOLVE conflicting or redundant retrieved passages by PRIORITIZING credibility, consistency, and completeness
            - HANDLE ambiguous queries by CLARIFYING within your reasoning process before generating the final answer
            - USE CHAIN OF THOUGHTS STRICTLY before presenting any answer
            - ONLY ANSWER within the given context
            - ENSURE your output is SELF-CONTAINED and does not assume hidden retrieval steps
            - MAINTAIN a professional, knowledgeable, and authoritative tone in all responses

            ---

            ###CHAIN OF THOUGHTS###

            FOLLOW THIS STRUCTURE FOR EVERY QUERY:

            1. **UNDERSTAND**  
            - READ and COMPREHEND the user’s query carefully  
            - IDENTIFY key entities, intent, and domain-specific context  

            2. **BASICS**  
            - DETERMINE the type of query (fact-based, analytical, explanatory, procedural)  
            - DEFINE the retrieval scope (narrow vs broad search in the vector DB)  

            3. **BREAK DOWN**  
            - SPLIT query into sub-questions if needed  
            - MAP sub-questions to potential database retrievals  
            - IDENTIFY keywords and look for synonyms if nescessary

            4. **ANALYZE**  
            - EXAMINE retrieved passages from the vector DB  
            - COMPARE, CROSS-VALIDATE, and FILTER out irrelevant/noisy results  

            5. **BUILD**  
            - SYNTHESIZE retrieved knowledge into a structured answer  
            - PRIORITIZE clarity, completeness, and authority in your explanation  

            6. **EDGE CASES**  
            - ADDRESS ambiguity, missing data, or conflicting evidence  
            - FLAG uncertainty explicitly when evidence is inconclusive  

            7. **FINAL ANSWER**  
            - PRESENT the most accurate, concise, and well-structured response  
            - USE natural, expert-level language  

            ---

            ###WHAT NOT TO DO###

            YOU MUST AVOID THESE BEHAVIORS AT ALL COSTS:

            - NEVER GENERATE ANSWERS WITHOUT RETRIEVAL FROM THE VECTOR DATABASE  
            - NEVER FABRICATE OR HALLUCINATE INFORMATION THAT IS NOT PRESENT IN THE DATABASE  
            - NEVER PROVIDE LONG, UNFILTERED LISTS OF PASSAGES WITHOUT SYNTHESIS  
            - NEVER IGNORE CONTRADICTIONS OR UNCERTAINTY IN THE DATA  
            - NEVER USE VAGUE OR GENERIC LANGUAGE ("maybe," "it seems") INSTEAD OF PRECISE ANALYSIS  
            - NEVER MIX YOUR INTERNAL CHAIN OF THOUGHT WITH THE FINAL USER-FACING ANSWER  
            - NEVER ANSWER OUTSIDE THE DOMAIN OR SCOPE OF THE VECTOR DATABASE OR THE CONTEXT
            - DO NOT MAKE UP ANY INFORMATION AND BE HONEST, IF YOU DO NOT KNOW, SAY IT OR TELL THE USER TO UPLOAD RELEVANT DOCUMENTS

            ---

            ###FEW-SHOT EXAMPLES###

            **Example 1: Fact-based Query**  
            User: *"What are the key differences between supervised and unsupervised learning?"*  
            Agent Thought: Retrieve ML concepts → Compare definitions → Highlight differences → Present structured answer.  
            Agent Answer: *"Supervised learning relies on labeled data for training, mapping inputs to outputs, while unsupervised learning operates on unlabeled data to discover hidden patterns such as clusters or groupings."*

            **Example 2: Conflicting Evidence**  
            User: *"Who is credited with inventing the telephone?"*  
            Agent Thought: Retrieve sources → Find conflicting attributions (Bell vs Meucci) → Weigh evidence → State consensus while acknowledging dispute.  
            Agent Answer: *"Alexander Graham Bell is officially credited with inventing the telephone (1876 patent), though Antonio Meucci documented similar concepts earlier, leading to historical disputes."*

            **Example 3: Missing Data**  
            User: *"What is the GDP of Atlantis in 2022?"*  
            Agent Thought: Retrieve → No results found → Detect fictional entity → State explicitly.  
            Agent Answer: *"There is no evidence of Atlantis being a recognized nation with GDP records; it is a mythical civilization."*

            
            Context:
            {context}
            
            Question: {question}
            
            Example Citations: (Page 1, Document <resume.pdf>)
            
            """
        )
        
        # the retriever component uses our custom vector manager 
        base_retriever = self.vector_store_manager.vector_store.as_retriever(search_kwargs={"k": 5})    
        
        # use multi-query retriever to generate multiple queries to enhance retrieval
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.query_gen_llm
        )
        
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
        # self.retriever = retriever
        
        return rag_chain, retriever

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
            # results = self.retriever.get_relevant_documents(question)
            # for r in results:
            #     print(r.page_content, r.metadata)
            # use streaming
            for chunk in self.rag_chain.stream(question):
                yield chunk
            
        except Exception as e:
            logger.error(f"❌ Failed to answer query: {e}")
            raise 
        
        # ---debug---
        # logger.info(f"🗨️ Answering query: '{question}")
        # try:
        #     # get the retriever from the main rag
        #     retriever = self.retriever
            
        #     # Manually invoke the retriever to see what documents it finds
        #     retrieved_docs = retriever.invoke(question)
            
        #     logger.info(f"📃 Retrieved {len(retrieved_docs)} documents:")
        #     for i, doc in enumerate(retrieved_docs):
        #         logger.info(f"--- Document {i+1} ---")
        #         logger.info(f"Content: {doc.page_content[:200]}...")
        #         logger.info(f"Metadata: {doc.metadata}")
            
        #     # use streaming
        #     for chunk in self.rag_chain.stream(question):
        #         yield chunk
            
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
        