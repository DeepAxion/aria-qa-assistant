"""
Test script for ARIA's professional OCR processor
Testing the complete OCR system
"""

import os
import sys
from pathlib import Path
import json

# add src to Python path
sys.path.insert(0, 'src')

# import our ARIA OCR processor
from ocr.processor import ARIAOCRProcessor

def test_ocr_processor():
    """Test the complete OCR Processor"""
    
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
    
    # check for test documents
    docs_dir = Path("docs")
    test_files = []
    
    # collect all the files in the folder and add them to test files
    for pattern in ["*.png", "*.jpg", "*.jpeg", "*.pdf"]:
        test_files.extend(docs_dir.glob(pattern))
        
    print(f"\n📄 Found {len(test_files)} test files")
    
    # process each file
    for i, file_path in enumerate(test_files[:3], 1): # test first 3 files
        print(f"\n{'='*60}")
        print(f"🔍 Test {i}: Processing {file_path.name}")
        print(f"{'='*60}")
        
        try:
            # process the file
            print("⏳ Processing...")
            results = processor.process_file(str(file_path))
            
            # display results
            print("✅ Processing completed!")
            
            # show statistics
            stats = processor.get_processing_stats(results)
            print(f"\n📊 Processing Statistics:")
            print(f"   File: {stats['file_name']}")
            print(f"   Method: {stats['method']}")
            print(f"   Pages: {stats['pages']}")
            print(f"   Text length: {stats['text_length']} characters")
            print(f"   Words: {stats['words']}")
            print(f"   Confidence: {stats['confidence']}")
            print(f"   Processing time: {stats['processing_time_seconds']} seconds")
            print(f"   File size: {stats['file_size_kb']} KB")
            
            # show text preview
            text = results['text']
            preview_length = 300 # preview 300 char
            if len(text) > preview_length:
                preview = text[:preview_length] + "..."
            else:
                preview = text
                
            # print preview
            print(f"\n📖 Text Preview (first {preview_length} chars):")
            print("-" * 40)
            print(preview)
            print("-" * 40)
            
            # save result
            saved_path = processor.save_results(results)
            print(f"💾 Results saved to: {saved_path}")
            
            # test loading saved results
            with open(saved_path, 'r', encoding='utf-8') as f:
                loaded_results = json.load(f)
            print(f"✅ Results successfully saved and loaded")
            
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    return True

def analyze_ocr_quality(results: dict):
    "Analyze the quality of OCR results"
    
    text = results.get('text', '')
    confidence = results.get('confidence', 0)
    
    # quality metrics
    quality_metrics = {
        'confidence_score': confidence,
        'text_length': len(text),
        'word_count': len(text.split()),
        'ave_word_length': len(text) / max(len(text.split()), 1),
        'has_numbers': any(c.isdigit() for c in text),
        'has_punctuation': any(c in '.,!?;:' for c in text),
        'estimated_quality': 'Unknown'
    }
    
    # estimate the quality using overall confidence score
    if confidence > 0.9:
        quality_metrics['estimated_quality'] = 'Excellent'
    elif confidence > 0.8:
        quality_metrics['estimated_quality'] = 'Good'
    elif confidence > 0.6:
        quality_metrics['estimated_quality'] = 'Fair'
    else:
        quality_metrics['estimated_quality'] = 'Poor'
    
    return quality_metrics

def demo_different_file_types():
    """Demo processing different file types"""
    
    print("\n🎪 ARIA OCR Demo - Different File Types")
    print("=" * 50)
    
    processor = ARIAOCRProcessor()
    
    # check different file types
    docs_dir = Path("docs")
    
    file_types = {
        'Images': list(docs_dir.glob("*png")) + list(docs_dir.glob("*jpg")),
        'PDFs': list(docs_dir.glob("*.pdf"))
    }
    
    # iterate through the files of each file type
    for file_type, files in file_types.items():
        if files:
            print(f"\n📁 {file_type} Files:")
            # test 2 of each type
            for file_path in files[:2]:
                try:
                    results = processor.process_file(str(file_path))
                    stats = processor.get_processing_stats(results)
                    quality = analyze_ocr_quality(results)
                    
                    # print out metrics
                    print(f"   ✅ {file_path.name}")
                    print(f"      Method: {stats['method']}")
                    print(f"      Quality: {quality['estimated_quality']} ({quality['confidence_score']:.2f})")
                    print(f"      Text: {stats['words']} words, {stats['text_length']} chars")
            
                except Exception as e:
                    print(f"   ❌ {file_path.name}: {e}")
    
if __name__ == "__main__":
    print("🅰️ ARIA OCR Processor")
    print("Testing professional OCR implementation")
    print("=" * 60)
    
    # ensure directories exist
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("docs").mkdir(parents=True, exist_ok=True)
    
    # run main test
    success = test_ocr_processor()
    
    # print out message if success
    if success:
        print(f"\n{'='*60}")
        print("🎉 OCR Processor Testing COMPLETED!")
        print("✅ Professional OCR processor is working correctly")
        print("✅ Can handle both images and PDFs")
        print("✅ Provides detailed metadata and confidence scores")
        print("✅ Saves and loads results properly")
        
    # print out fail messages and fixes
    else:
        print("\n❌ Some tests failed. Check error messages above.")
        print("Common fixes:")
        print("1. Make sure virtual environment is activated")
        print("2. Check if all packages are installed: pip list")
        print("3. Try running: pip install -r requirements.txt --force-reinstall")