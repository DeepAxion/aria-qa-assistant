"""Test chunker"""

import os
import sys
from pathlib import Path
import json

# add src to Python path
sys.path.insert(0, 'src')

# import our ARIA OCR processor
from ocr.processor import ARIAOCRProcessor
from ocr.chunker import ARIAChunker


def test_chunker():
    """Demonstrates the chunking process on a processed document."""
    print("\n📦 ARIA Chunker Demo - Text Chunking Process")
    print("=" * 50)
    
    # initialize processor
    print("🚀 Initializing ARIA OCR Processor...")
    try:
        processor = ARIAOCRProcessor(
            lang='en',
            use_gpu=False,
            confidence_threshold=0.6
        )
        print("✅ OCR processor initialized successfully!")
    except Exception as e:
        print(f"😟 Falied to initialize processor: {e}")
        return False
    
    # initialize chunker
    print("🚀 Initializing ARIA OCR Processor...")
    try:
        chunker = ARIAChunker(chunk_size=500, chunk_overlap=50)
        print("✅ Chunker initialized successfully!")
    except Exception as e:
        print(f"😟 Falied to initialize processor: {e}")
        return False

    # check test documents
    docs_dir = Path("docs")   
    test_files = []
     
    # collect all the files in the folder and add them to test files
    for pattern in ["*.png", "*.jpg", "*.jpeg", "*.pdf"]:
        test_files.extend(docs_dir.glob(pattern))
        
    print(f"\n📄 Found {len(test_files)} test files")
    
    # iterate through the files
    for i, file_path in enumerate(test_files, 1): # test first 3 files
        print(f"\n{'='*60}")
        print(f"🔍 Test {i}: Processing {file_path.name}")
        print(f"{'='*60}")
        
        try:
            # step 1: process the file with ocr processor
            print("⏳ Processing...")
            ocr_results = processor.process_file(str(file_path))
            
            # step 2: pass the extracted text and metadata to the chunker
            print("⏳ Chunking the extracted text...")
            chunks = chunker.chunk_text(ocr_results['text'], ocr_results)
            # display results
            print(f"✅ Successfully created {len(chunks)} chunks.")
            
            # display a preview of the first and last chunk
            if chunks:
                print("\n📖 Chunk Preview:")
                print("-" * 40)

                # first chunk
                first_chunk = chunks[0]
                print(f"**First Chunk (Page {first_chunk.get('page_number', 'N/A')})**: {first_chunk['text'][:150]}...")
                
                # last chunk
                last_chunk = chunks[-1]
                print(f"**Last Chunk (Page {first_chunk.get('page_number', 'N/A')})**: {last_chunk['text'][:150]}...")
                
                print("-" * 40)
            
        except Exception as e:
            print(f"❌ An error occured in chunking process: {e}")
            import traceback
            traceback.print_exc()
            
    return True


if __name__ == "__main__":
    print("🎵 ARIA OCR Processor and Chunker Test Suite")
    print("Testing professional OCR implementation and document chunking")
    print("=" * 60)
    
    # Ensure directories exist
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("docs").mkdir(parents=True, exist_ok=True)
    
    # Run chunking process
    success = test_chunker() 
    
    # print out message if success
    if success:
        print(f"\n{'='*60}")
        print("🎉 Chunker Testing COMPLETED!")
        
    # print out fail messages and fixes
    else:
        print("\n❌ Some tests failed. Check error messages above.")
        print("Common fixes:")
        print("1. Make sure virtual environment is activated")
        print("2. Check if all packages are installed: pip list")
        print("3. Try running: pip install -r requirements.txt --force-reinstall")
                
    
    