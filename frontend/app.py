import streamlit as st
import requests

# define the base URL of your FastAPI backend
# use the local URL for development
FASTAPI_BACKEND_URL = "http://127.0.0.1:8000"

# set up the Streamlit page
st.set_page_config(page_title="ARIA Document Q&A Assistant")
st.title("🅰️ ARIA Document Q&A Assistant")

# --- Document Upload Section ---
st.header("📃 Upload a document")
uploaded_file = st.file_uploader("Choose a PDF or image file", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    # add a button to manually start processing
    process_button = st.button("Process file")
    # only run if the button is clicked
    if process_button:
        try:
            # use st.spinner to show a progress indicator
            with st.spinner("Processing document... Please wait."):
                # prepare the file for upload
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                ### make a POST request to the /upload endpoint of your FastAPI backend
                response = requests.post(f"{FASTAPI_BACKEND_URL}/upload", files=files)
                
                if response.status_code == 200:
                    st.success("Document processed successfully!")
                    st.session_state.is_document_processed = True
                else:
                    st.error(f"Error processing document: {response.json().get('error', 'Unknown error')}")
                    st.session_state.is_document_processed = False
                    
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

# --- Query Section ---
st.header("💭 Ask a Question")
# a simple way to check if a document has been uploaded before enabling the text input
if 'is_document_processed' not in st.session_state:
    st.session_state.is_document_processed = False

question = st.text_input(
    "Enter your question here:",
    # disabled=not st.session_state.is_document_processed,
    placeholder="e.g., What is the project's tech stack?"
)

# use a button to submit the query
if st.button("Get Answer", disabled= not question):
    with st.spinner("Searching for an answer..."):
        # Make a POST request to the /query endpoint
        try:
            response = requests.post(
                f"{FASTAPI_BACKEND_URL}/query",
                json={"question": question},
                stream=True # Use stream=True to handle the streaming response
            )
            
            if response.status_code == 200:
                # Use st.empty to update the response as it streams
                answer_placeholder = st.empty()
                full_answer = ""
                for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                    full_answer += chunk
                    answer_placeholder.write(full_answer)
                
            else:
                st.error(f"Error getting answer: {response.json().get('error', 'Unknown error')}")
                
        except requests.exceptions.ConnectionError as e:
            st.error("Could not connect to the FastAPI backend. Please ensure the backend server is running.")