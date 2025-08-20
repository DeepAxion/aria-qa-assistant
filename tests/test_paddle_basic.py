from paddleocr import PaddleOCR
import cv2
import numpy as np
from PIL import Image
import os


def test_paddleocr():
    '''Function to load paddleOCR'''
    print("🎵Testing PaddleOCR for ARIA...")
    
    # initialize paddleOCR
    # this downloads models on the first run 
    print("Loading PaddleOCR models...")
    # loading ocr pretrained model, enabling angle classification (detecting rotated text )
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False) # paddleOCR already takes care of caching models
    print("✅PaddleOCR loaded successfully")
    
    return ocr

def test_with_simple_image(ocr):
    """Test with a simple text image"""
    # create a simple test image with text
    # this helps verify everything is working
    img = Image.new('RGB', (200, 200), color='white') # random 200x200 image
    
    # test without drawing text first, just to see if OCR loads
    img_array = np.array(img)
    
    print(" Testing OCR on blank image...")
    result = ocr.ocr(img_array, cls=True)
    print("✅ OCR test completed!")
    
    return result

if __name__ == "__main__":
    try:
        # test 1: Load OCR
        ocr_engine = test_paddleocr()
        
        # test 2: Run on a simple image
        result = test_with_simple_image(ocr_engine)
        print("Result:", result)
        
        print("🎉 PaddleOCR is working correctly!")
        print("Ready to process real documents! Let's go")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Check if you activated virtual environment: venv\\Scripts\\activate")