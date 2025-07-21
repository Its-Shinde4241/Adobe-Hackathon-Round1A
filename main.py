#!/usr/bin/env python3
import os
import json
import re
import sys
from pathlib import Path
import fitz  # PyMuPDF
from typing import List, Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFOutlineExtractor:
    def _init_(self):
        # Font size thresholds for heading detection
        self.title_min_size = 16
        self.h1_min_size = 14
        self.h2_min_size = 12
        self.h3_min_size = 10

        # Common heading patterns
        self.heading_patterns = [
            r'^\d+\.\s+',  # 1. Chapter
            r'^\d+\.\d+\s+',  # 1.1 Section
            r'^\d+\.\d+\.\d+\s+',  # 1.1.1 Subsection
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',  # Title Case
        ]

        # Words that commonly appear in headings
        self.heading_keywords = {
            'introduction', 'conclusion', 'abstract', 'summary', 'overview',
            'background', 'methodology', 'results', 'discussion', 'references',
            'chapter', 'section', 'appendix', 'acknowledgments', 'contents'
        }

    def extract_text_with_formatting(self, doc: fitz.Document) -> List[Dict]:
        """Extract text with font information from PDF"""
        text_blocks = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")

            for block in blocks.get("blocks", []):
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            text_blocks.append({
                                "text": text,
                                "size": span["size"],
                                "flags": span["flags"],
                                "font": span["font"],
                                "page": page_num + 1,
                                "bbox": span["bbox"]
                            })

        return text_blocks

    def is_likely_heading(self, text: str, font_size: float, flags: int) -> bool:
        """Determine if text is likely a heading based on multiple criteria"""
        text_lower = text.lower()

        # Check for common heading patterns
        for pattern in self.heading_patterns:
            if re.match(pattern, text):
                return True

        # Check for heading keywords
        words = text_lower.split()
        if any(keyword in words for keyword in self.heading_keywords):
            return True

        # Font-based checks
        is_bold = flags & 2**4  # Bold flag
        is_large = font_size > self.h3_min_size

        # Length-based filtering (headings are usually not too long)
        if len(text) > 200:
            return False

        # Combined criteria
        if is_bold and is_large:
            return True

        if font_size > self.h1_min_size and len(text) < 100:
            return True

        return False

    def classify_heading_level(self, text: str, font_size: float, all_sizes: List[float]) -> str:
        """Classify heading level based on font size and content"""
        text_lower = text.lower()

        # Sort unique font sizes in descending order
        unique_sizes = sorted(set(all_sizes), reverse=True)

        # Title detection (largest font, usually on first few pages)
        if font_size >= self.title_min_size and font_size == max(all_sizes):
            return "TITLE"

        # Pattern-based classification
        if re.match(r'^\d+\s+', text) or re.match(r'^chapter\s+\d+', text_lower):
            return "H1"
        elif re.match(r'^\d+\.\d+\s+', text):
            return "H2"
        elif re.match(r'^\d+\.\d+\.\d+\s+', text):
            return "H3"

        # Font size-based classification
        if len(unique_sizes) >= 3:
            if font_size == unique_sizes[0]:
                return "H1"
            elif font_size == unique_sizes[1]:
                return "H2"
            elif font_size == unique_sizes[2]:
                return "H3"

        # Fallback based on absolute font sizes
        if font_size >= self.h1_min_size:
            return "H1"
        elif font_size >= self.h2_min_size:
            return "H2"
        else:
            return "H3"

    def extract_title(self, text_blocks: List[Dict]) -> str:
        """Extract document title"""
        # Look for title in first few pages
        first_page_blocks = [block for block in text_blocks if block["page"] <= 3]

        if not first_page_blocks:
            return "Untitled Document"

        # Find the largest font size on first pages
        max_size = max(block["size"] for block in first_page_blocks)

        # Get text with largest font size
        title_candidates = [
            block["text"] for block in first_page_blocks
            if block["size"] == max_size and len(block["text"].strip()) > 3
        ]

        if title_candidates:
            # Return the longest candidate (likely the actual title)
            title = max(title_candidates, key=len).strip()
            # Clean up title
            title = re.sub(r'\s+', ' ', title)
            return title

        return "Untitled Document"

    def process_pdf(self, pdf_path: str) -> Dict:
        """Process a PDF and extract outline"""
        try:
            logger.info(f"Processing PDF: {pdf_path}")

            doc = fitz.open(pdf_path)
            text_blocks = self.extract_text_with_formatting(doc)

            if not text_blocks:
                logger.warning(f"No text found in {pdf_path}")
                return {"title": "Empty Document", "outline": []}

            # Extract title
            title = self.extract_title(text_blocks)

            # Get all font sizes for heading classification
            all_sizes = [block["size"] for block in text_blocks]

            # Find potential headings
            potential_headings = []
            for block in text_blocks:
                if self.is_likely_heading(block["text"], block["size"], block["flags"]):
                    level = self.classify_heading_level(block["text"], block["size"], all_sizes)
                    if level in ["H1", "H2", "H3"]:
                        potential_headings.append({
                            "level": level,
                            "text": block["text"].strip(),
                            "page": block["page"],
                            "size": block["size"]
                        })

            # Remove duplicates while preserving order
            seen = set()
            outline = []
            for heading in potential_headings:
                key = (heading["text"], heading["page"])
                if key not in seen:
                    seen.add(key)
                    outline.append({
                        "level": heading["level"],
                        "text": heading["text"],
                        "page": heading["page"]
                    })

            # Sort by page number
            outline.sort(key=lambda x: x["page"])

            doc.close()

            result = {
                "title": title,
                "outline": outline
            }

            logger.info(f"Extracted {len(outline)} headings from {pdf_path}")
            return result

        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return {"title": "Error Processing Document", "outline": []}

def main():
    """Main function to process all PDFs in input directory"""
    # Check if running in Docker or locally
    if os.path.exists("/app/input"):
        input_dir = Path("/app/input")
        output_dir = Path("/app/output")
    else:
        # Local testing mode
        input_dir = Path("./input")
        output_dir = Path("./output")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        print(f"Please create the '{input_dir}' directory and place PDF files in it.")
        sys.exit(1)

    extractor = PDFOutlineExtractor()

    # Find all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in input directory")
        print(f"No PDF files found in {input_dir}")
        print(f"Please place PDF files in the '{input_dir}' directory.")
        return

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    # Process each PDF
    for pdf_file in pdf_files:
        try:
            result = extractor.process_pdf(str(pdf_file))

            # Generate output filename
            output_file = output_dir / f"{pdf_file.stem}.json"

            # Save result
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved outline to {output_file}")
            print(f"✓ Processed: {pdf_file.name} → {output_file.name}")

        except Exception as e:
            logger.error(f"Failed to process {pdf_file}: {str(e)}")
            print(f"✗ Failed: {pdf_file.name}")

if __name__ == "_main_":
    main()