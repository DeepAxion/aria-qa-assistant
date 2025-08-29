from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from src.retrieval.rag_pipeline import ARIARAGPipeline
from pydantic import BaseModel
import os
import shutil
import sys

# add the parent directory to the Python path
# to allow the app to find the ocr and embeddings modules
# sys.path.insert(0, str(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))))

# Initialize the RAG Pipeline instance
# this is done whenever the app starts
rag_pipeline = ARIARAGPipeline()

# initialize fastapi
app = FastAPI(
    title="ARIA PROJECT",
    description="API for ARIA, a RAG pipeline for document Q&A",
    version="1.0.0"
)

# Define the Root endpoint
@app.get("/")
async def read_root():
    return {"message": "ARIA is up and running. Go to /docs  for API documentation."}

# Define the Upload endpoint
@app.post("/upload")
async def upload_document(file: UploadFile=File(...)):
    """Upload and process documents"""
    try:
        # save the file to a tempt dir
        os.makedirs("temp", exist_ok=True)
        file_path = f"temp/{file.filename}"
        
        # save the UploadFile object to disk
        # as the rag_pipeline takes in a valid string path from disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # ingest the document
        rag_pipeline.ingest_document(file_path)
        
        # clean up the temporary file
        os.remove(file_path)
        
        return {"filename": file.filename, "status": "Document ingested successfully!"}
    
    except Exception as e:
        return {"error": str(e), "status": "Document ingestion failed:(("}
    

# Define a Pydantic model for the request body
# a schema that defines the expected structure and data types of the incoming request
class QueryRequest(BaseModel):
    question: str
    
@app.post("/query")
async def answer_query(query: QueryRequest):
    """Answer the question based on ingested documents"""
    try:
        # get the answer from RAG pipeline
        answer_generator = rag_pipeline.answer_query(query)
        
        # return streamed chunks
        return StreamingResponse(answer_generator, media_type="text/plain")    
    except Exception as e:
        return {"error": str(e)}
    



