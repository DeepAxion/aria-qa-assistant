from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_test_doc():
    """Create a test document image"""
    
    # create image with sample text
    img = Image.new('RGB', (600, 400), color='white')
    # use draw to write content
    draw = ImageDraw.Draw(img)
    
    # sample business doc
    text_lines = [
        "BUSINESS REPORT",
        "",
        "Company: ARIA Technologies",
        "Date: December 2024",
        "Revenue: $1,500,000",
        "Profit Margin: 15.7%",
        "",
        "Key Metrics:",
        "- Customer Growth: 23%",
        "- Employee Count: 45",
        "- Market Share: 12.3%"
    ]
    
    # draw text
    # define start position 
    y_position = 20 
    for line in text_lines:
        # we skip empty lines
        if line.strip():
            # x_pos = 20 (20 pixels from the left), y_postion space from top 
            draw.text((20, y_position), line, fill='black')
        y_position += 30 # increment line spacing
    
    img.save("test_doc.png")
    print("✅ Created test document: test_document.png")
    
    return "test_doc.png"
    
def test_ocr_with_text():
    '''Test OCR with actual text document'''
    print("🎵 Testing ARIA OCR with real text...")
    
    # create test doc
    test_image = create_test_doc()
    
    # initiate OCR model
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    
    # process document
    print("Processing the document...")
    result = ocr.ocr(test_image, cls=True)
    
    # display result
    print("\n📄 OCR Results: ")
    print("-" * 40)
    
    '''
    What result looks like:
    [
        [  # first image (since you can process multiple images at once)
            [
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], line[0]  # Bounding box (quadrilateral) -> default top left/right and bottom left/right
            [recognized_text, confidence_score]    line[1]   # Text and confidence
            ],
            ...
        ]
    ]
    '''
    if result and result[0]:
        # iterate through the lines in result
        for line in result[0]:
            # extract the text
            text = line[1][0] 
            # get the confidence
            confidence = line[1][1]
            print(f"Text: {text}")
            print(f"Confidence: {confidence:.2f}")
            print("-" * 20)
    else:
        print("No text detected")

    return result
    
if __name__== "__main__":
    test_ocr_with_text()