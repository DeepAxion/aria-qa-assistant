"""
ARIA OCR Processor - Day 3
Professional OCR processing class for document analysis
"""

import os
import re
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import time

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from langchain_core.documents import Document
import pypdf


# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )

# Ensure logs directory exists
Path("data/logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data/logs/ocr.log", mode="w", encoding="utf-8"),  # overwrite each run
        logging.StreamHandler()  # also print to console
    ],
    force=True  # ensures config applies even if logging was set before
)

logger = logging.getLogger(__name__)


# OCR class
class ARIAOCRProcessor:
    """
    Professional OCR processor for ARIA document analysis
    Handles both images and pdfs with intelligent text extraction
    """
    
    # building a constructor
    def __init__(self, 
                 lang: str= 'en',
                 use_gpu: bool = False,
                 confidence_threshold: float = 0.5):
        
        """
        Initialize ARIA OCR processor
        
        Args:
            lang: Language for OCR (default: 'en')
            use_gpu: Whether to use GPU acceleration
            confidence_threshold: Minimum confidence for text acceptance
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        
        # print message
        logger.info("😃 Initializing ARIA OCR Processor...")
        logger.info(f"Language: {lang}, GPU usage: {use_gpu}, Confidence Threshold: {confidence_threshold}")
        
        # initialize the OCR instance
        try:
            # start timer
            start_time = time.time()
            
            self.ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                use_gpu=use_gpu,
                show_log=False
            )
            
            # stop timer
            stop_time = time.time()
            # calculate load time
            load_time = stop_time - start_time
            #print successfull message
            logger.info(f"✅ OCR Engine initialized successfully in {load_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to initialize OCR Engine: {e}")
            raise # rethrow error
    
    def process_file(self, file_path: str) -> Dict:
        """
        Process any supported file (PDF, image) and extract text
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Dictionary with extracted text and metadata
        """
        
        # turn file_path into Path object to inspect and manipulate the path
        file_path = Path(file_path)
        
        # check if it exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"📄 Processing file: {file_path}")
        
        # know what we're dealing with: get file extension to decide how to process it
        file_extension = file_path.suffix.lower() # lowercase
        
        # PDF
        if file_extension == '.pdf':
            return self._process_pdf(file_path)
        # PNG, JPEG, JPG, BMP
        elif file_extension in ['.png', '.jpg', '.jpeg', '.bmp']:
            return self._process_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def _process_image(self, image_path: str) -> Dict:
        """Process single image file"""
        
        logger.info(f"🖼️ Processing image: {image_path.name}")
        
        try: 
            # start timer
            start_time = time.time()
            
            # load and covert to numpy
            image = Image.open(image_path) # PIL image
            img_array = np.array(image)
            
            # print image info
            logger.info(f"Image size: {image.size}, Mode: {image.mode}")
            
            # run ocr
            ocr_result = self.ocr_engine.ocr(img_array, cls=True)
            
            # process result
            text, confidence = self._process_page_ocr_result(ocr_result)
            
            processing_time = time.time() - start_time
            
            # add page header, page 1 because there is only 1 page
            page_header = f"---Page 1---\n"
            text = page_header + text
            
            logger.info(f"✅ Image processed in {processing_time:2f} seconds")
            
            # return result dict
            return self._create_result_dict(
                text=text,
                method="ocr_image",
                file_path=image_path,
                confidence=float(confidence),
                total_pages=1,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"😟 Image processing failed: {e}")
            raise
    
    def _process_pdf(self, pdf_path: str) -> Dict:
        """Process PDF file by directly extracting the pdf, if failed, fallback to OCR"""
        
        logger.info(f"🔍 Processing PDF: {pdf_path.name}")
        
        # Step 1: Try direct extraction first (for text-based pdf only)
        direct_text = self._extract_text_from_pdf_direct(pdf_path)
        
        # Step 2: If direct extraction gives good result -> use it
        if self._is_text_extraction_good(direct_text):
            logger.info("✅ Using direct PDF text extraction")
            return self._create_result_dict(
                text=direct_text,
                method='direct_pdf',
                file_path=pdf_path,
                confidence=1.0,
                total_pages=self._count_pdf_pages(pdf_path)
            )
        
        # Step 3: Fallback to OCR
        logger.info("🤖 Direct extraction failed. Switching to OCR Processing...")
        return self._process_pdf_with_ocr(pdf_path)
    
    def _extract_text_from_pdf_direct(self, pdf_path: str) -> str:
        """Extract text from pdf directly using pypdf
        We want to return this
        text_part = [
            "--- Page 1 ---Hello, this is page one.",
            "--- Page 2 ---\nThis is page two with some content.",
            "--- Page 3 ---\nConclusion and closing remarks."
        ]
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text_parts = []
                
                # iterate through each page of the document, starting from page 1
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    # extract the text of that page
                    page_text = page.extract_text()
                    # clean the text
                    clean_page_text = self._clean_text(page_text)
                    
                    # if not whitesapce or empty line, add to text_parts
                    if page_text.strip():
                        text_parts.append(f"---Page {page_num}--- \n{clean_page_text}")
                # separate every element with 2 lines
                return '\n\n'.join(text_parts)
        except Exception as e:
            logger.warning(f"Direct PDF Extraction failed: {e}")
            return ""
    
    def _is_text_extraction_good(self, text: str) -> bool:
        """Checking if direct extraction yields good results"""
        # if the text is empty or lenght of text < a sentence (30 characters) -> return False
        if not text or len(text.strip()) < 30:
            return False
        
        # check if there are less than 10 words
        words = text.split()
        if len(words) < 10:
            return False
        
        # check if there are too many weird characters (sign of OCR artifacts)
        weird_char = sum(1 for c in text if ord(c) > 127)
        if weird_char / len(text) > 0.1: # if there is more than 10% of those
            return False
        
        return True
    
    def _count_pdf_pages(self, pdf_path: str) -> int:
        """Counting pages in pdf"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(pdf_path)
                # count elements in the list of pages 
                return len(pdf_reader.pages)
                
        except Exception as e:
            logger.warning(f"Cannot count PDF pages: {e}")
            return 1
        
    def _process_pdf_with_ocr(self, pdf_path: str) -> Dict:
        """Process PDF using OCR (for image-based pdfs)"""
        try:
            # start timer
            start_time = time.time()
            
            # convert PDF into IMAGEs, OCR extracts text from images
            logger.info("📄->🖼️ Converting PDF to images...")
            images = convert_from_path(
                str(pdf_path),
                dpi=200, # good balance of quality vs speed
                first_page=1, # process from first page
                last_page=None, # process all, skip none
                thread_count=1 # single thread for stability
            )
            
            logger.info(f"📋 Processing {len(images)} pages...")
            
            # now we process each page
            all_pages_data = []
            full_text_parts = []
            
            # iterate through the list of images
            for page_num, image in enumerate(images, 1):
                logger.info(f"🔍 Processing page {page_num}/{len(images)}")
                
                # convert PIL image to numpy array
                img_array = np.array(image)
                
                # feed to image array to ocr engine
                page_result = self.ocr_engine.ocr(img_array, cls=True)
                
                # process the ocr result in this page, calling the process_page_ocr_result
                page_text, page_confidence = self._process_page_ocr_result(page_result)
                
                # store page result
                page_data = {
                    'page_number': page_num,
                    'text': page_text,
                    'confidence': page_confidence,
                    'word_count': len(page_text.split()),
                    'char_count': len(page_text)
                }
                
                # append paragraphs into page list
                all_pages_data.append(page_data)
                full_text_parts.append(f"---Page {page_num}--- \n{page_text}")
                
                # log progress
                logger.info(f"✅ Page {page_num}: {len(page_text)} chars, {page_confidence:.2f} confidence")
                
            # combine all paragraphs
            full_text = '\n\n'.join(full_text_parts)
            overall_confidence = np.mean([page['confidence'] for page in all_pages_data])
            
            # time
            processing_time = time.time() - start_time
            logger.info(f"🥳 PDF OCR processing completed in {processing_time:.2f} seconds")
            
            # return result dict
            return self._create_result_dict(
                text=full_text,
                method='ocr_pdf',
                file_path=pdf_path,
                confidence=overall_confidence,
                total_pages=len(images),
                pages_data=all_pages_data,
                processing_time=processing_time
            )
                
        except Exception as e:
            logger.error(f"😟 PDF OCR Processing failed: {e}")
            raise   
        
    def _process_page_ocr_result(self, ocr_result: list) -> Tuple[str, float]:
        """
        Process OCR result from a single page
        
        Args:
            ocr_result: Raw OCR result from PaddleOCR
            Example of raw ocr result
            page_text = [
                [  # first image (since you can process multiple images at once)
                    [
                    [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], line[0]  # Bounding box (quadrilateral) -> default top left/right and bottom left/right
                    [recognized_text, confidence_score]    line[1]   # Text and confidence
                    ],
                    ...
                ]
            ]
            
        Returns:
            Tuple of (cleaned_text, average_confidence)
        """
        
        # check if valid
        if not ocr_result or not ocr_result[0]:
            return "", 0.0
        
        text_lines = []
        confidences = []
        
        # extract text and confidence from the 
        # ocr_result[0] gets the first (and usually only) page of results
        for detection in ocr_result[0]:
            # check if detection is valid: has at least 2 parts and the text part is not empty
            if len(detection) >= 2 and detection[1]:
                text = detection[1][0]
                confidence = detection[1][1]
                
                # only include text above the confidence threshold
                if confidence >= self.confidence_threshold and text.strip():
                    text_lines.append(text.strip())
                    confidences.append(confidence)
                    
        # combine clean text into paragraphs
        text = self._reconstruct_text_structure(text_lines)
        # clean text 
        clean_text = self._clean_text(text)
        
        # calculate the average confidence
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return clean_text, avg_confidence

    def _reconstruct_text_structure(self, text_lines: List[str]) -> str:
        """Reconstruct proper text structure from OCR lines
            text_lines = [
                            "This is the first line of a paragraph",
                            "that continues here.",
                            "Now a new paragraph starts and",
                            "it ends here."
                        ]
            after reconstruction:
            [
                "This is the first line of a paragraph that continues here.",
                "Now a new paragraph starts and it ends here."
            ]
        """
        if not text_lines:
            return ""
        
        # simple approach: join lines with spaces, create paragraphs on sentence endings
        paragraphs = []
        current_paragraph = []
        
        # iterate through the line
        for line in text_lines:
            line = line.strip()
            if not line: 
                continue
            
            current_paragraph.append(line)
            
            # end paragraph on sentence ending punctuation
            if line.endswith(('.', '!', '?', ':')):
                if current_paragraph:
                    # join current paragraph
                    # add to paragraph
                    paragraphs.append(' '.join(current_paragraph))
                    # empty the buffer
                    current_paragraph = []
        
        # if there is any remaining text, probably doesn't end with those punctuation
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
            
        return '\n\n'.join(paragraphs)
    
    def _clean_text(self, text: str):
        """
        Performs basic text cleaning and normalization.
        Removes artifacts, handles special characters, and normalizes spacing.
        """
        
        if not text:
            return ""
        
        # lowercase first
        text = text.lower()
        
        # replace multiple new lines with a single one to maintain paragraphs
        text = re.sub(r'\n\s*\n', '\n\n', text)

        # remove common OCR artifacts and unwanted characters
        # example: removes non-alphanumeric except for punctuation and spaces
        text = re.sub(r'[^a-z0-9\s@$%+.,?!;:\-()\'\"]', '', text)
        
        # replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    def _create_result_dict(self, 
                           text: str, 
                           method: str, # what method do we use
                           file_path: Path, 
                           confidence: float,
                           total_pages: int,
                           pages_data: List[Dict] = None,
                           processing_time: float = 0.0) -> Dict:
        
        """Create a standardized result dict"""
        return {
            # content
            'text': text,
            'text_length': len(text),
            'word_count': len(text.split()),
            
            # metadata
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_size_bytes': file_path.stat().st_size,
            'file_extension': file_path.suffix.lower(),
            
            # processing info
            'processing_method': method,
            'confidence': confidence,
            'total_pages': total_pages,
            'processing_time': processing_time,
            
            # detailed page data if available
            'pages': pages_data or [],
            
            # system info
            'ocr_engine': 'PaddleOCR',
            'language': self.lang,
            'confidence_threshold': self.confidence_threshold,
            
            # timestamp
            'timestamp': time.time()
        }
        
    def save_results(self, results: Dict, output_dir: str = "data/processed")->str:
        """
        Save OCR results to JSON file
        
        Args:
            results: OCR results dictionary
            output_dir: Directory to save results
            
        Returns:
            Path to saved file
        """
        # create path object
        output_dir = Path(output_dir)
        # output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # create filename based on original file
        # document.pdf => document_ocr_results.json
        original_name = Path(results["file_name"]).stem 
        output_filename = f"{original_name}_ocr_results.json"
        # combine the dir name and file name
        output_path = output_dir/output_filename
        
        # save with nice formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Results saved to: {output_path}")
        return str(output_path)
    
    def get_processing_stats(self, results: Dict) -> Dict:
        """Extract processing statistics from results, which is a dictionary"""
        
        stats = {
            'file_name': results.get('file_name', 'Unknown'),
            'method': results.get('processing_method', 'Unknown'),
            'pages': results.get('total_pages', 1),
            'text_length': results.get('text_length', 0),
            'words': results.get('word_count', 0),
            'confidence': f"{results.get('confidence', 0):.2f}",
            'processing_time_seconds': f"{results.get('processing_time', 0):.2f}",
            'file_size_kb': f"{results.get('file_size_bytes', 0)/1024:.1f}"
        }
        
        return stats
    
    def get_langchain_documents(self, file_path: str) -> List[Document]:
        """
        Process a file and return a list of Langchain Document objects.
        
        Args:
            file_path: Path to the file to process.
            
        Returns:
            List of Document objects with content and metadata.
        """
        
        # process the file and get the result dictionary
        results = self.process_file(file_path)
        
        # extract the text and page-wise data
        full_text = results.get('text', '')
        pages_data = results.get('pages_data', [])
        
        if not full_text:
            logger.warning(f"No text extracted from the file: {file_path}")
            return []
        
        documents = []
        
        # check if we have page level data
        if pages_data:
            # create a langchain document for each page
            for page in pages_data:
                doc_metadata = {
                    "sources": results['file_name'],
                    "page": page['page_number'],
                    "confidence": page["confidence"],
                    "method": results['processing_method']
                }
                documents.append(Document(page_content=page['text'], metadata=doc_metadata))
                
        else:
            # if no pages_data, create a single document for an entire text
            doc_metadata = {
                "sources": results['file_name'],
                "confidence": results['confidence'],
                "method": results['processing_method']
            }
            documents.append(Document(page_content=full_text, metadata=doc_metadata))
        
        logger.info(f"Created {len(documents)} Langchain Document objects.")
        return documents
        
        