import os
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import logging
import time

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Error: PyMuPDF not installed. Install with: pip install PyMuPDF")
    sys.exit(1)

# Configure logging with UTF-8 support
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

@dataclass
class TextBlock:
    """Text block with metadata"""
    text: str
    size: float
    flags: int
    font: str
    page: int
    bbox: Tuple[float, float, float, float]
    is_bold: bool = False
    is_italic: bool = False

class PDFOutlineExtractor:
    """Optimized PDF outline extractor for Round 1A with enhanced multilingual support"""

    def __init__(self):
        # Stricter thresholds for better precision
        self.min_heading_size_ratio = 1.15  # Relative to average text size
        self.max_heading_length = 150
        self.min_heading_length = 2  # Reduced for CJK characters

        # Enhanced heading patterns with better CJK support
        self.heading_patterns = [
            # Numbers with dots/spaces (strict)
            r'^\d+\.?\s+[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ffA-Z]',  # Include CJK
            r'^\d+\.\d+\.?\s+[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ffA-Z]',
            r'^\d+\.\d+\.\d+\.?\s+[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ffA-Z]',

            # Chapter/Section patterns (multilingual)
            r'^(Chapter|CHAPTER|章|第.*章)\s*\d*',
            r'^(Section|SECTION|節|第.*節)\s*\d*',
            r'^(Part|PART|部|第.*部)\s*[IVX\d]*',

            # Roman numerals with proper format
            r'^[IVX]{1,4}\.?\s+[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ffA-Z]',

            # Letters with dots (must be followed by text)
            r'^[A-Z]\.?\s+[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ffA-Z]',

            # Parentheses with text following
            r'^\(\d+\)\s*[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ffA-Z]',
            r'^\([A-Za-z]\)\s*[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ffA-Z]',

            # Enhanced CJK patterns
            r'^第[一二三四五六七八九十百千万\d]+[章节節條項部分]\s*',  # Chinese/Japanese chapters
            r'^[一二三四五六七八九十百千万]+[、\s]*[\u4e00-\u9fff]+',  # Chinese numbers with text
            r'^[\u4e00-\u9fff]{1,3}[、．。]\s*[\u4e00-\u9fff]',  # CJK with punctuation
            r'^第[\d]+[章节節條項部分]',  # Numbered chapters in CJK

            # Arabic patterns
            r'^الفصل\s+\d+',
            r'^القسم\s+\d+',

            # Japanese specific
            r'^[１２３４５６７８９０]+[章節項目]\s*',  # Full-width numbers
            r'^[あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん]{1,3}[。、]\s*',  # Hiragana
        ]

        # Expanded multilingual heading keywords
        self.heading_keywords = {
            # English
            'introduction', 'conclusion', 'abstract', 'summary', 'overview',
            'background', 'methodology', 'methods', 'results', 'discussion',
            'references', 'bibliography', 'appendix', 'acknowledgments',
            'preface', 'foreword', 'contents', 'glossary', 'index',
            'chapter', 'section', 'part', 'unit', 'lesson',

            # Chinese (Simplified & Traditional)
            '概要', '要約', '序論', '背景', '手法', '結果', '考察', '結論',
            '参考文献', '付録', '目次', '前書き', '謝辞', 'はじめに',
            '摘要', '简介', '背景', '方法', '结果', '讨论', '结论',
            '参考文献', '附录', '目录', '前言', '致谢', '开始',
            '緒論', '方法論', '結果', '討論', '結論', '參考文獻',
            '附錄', '目錄', '前言', '致謝', '開始', '章', '節', '部',

            # Japanese
            '序論', '序章', '緒言', '概要', '要約', '背景', '手法', '方法',
            '結果', '考察', '結論', '謝辞', '参考文献', '付録', '目次',
            'はじめに', 'おわりに', '章', '節', '項', '部', '編',

            # Arabic
            'مقدمة', 'خلاصة', 'ملخص', 'نتائج', 'مناقشة', 'خاتمة',
            'مراجع', 'ملحق', 'فهرس', 'تمهيد', 'شكر', 'فصل',

            # Other languages
            'einführung', 'zusammenfassung', 'ergebnisse', 'diskussion',
            'introducción', 'resumen', 'resultados', 'discusión',
            'введение', 'заключение', 'результаты', 'обсуждение',
        }

        # Patterns to exclude (enhanced for multilingual)
        self.exclusion_patterns = [
            # Page numbers and references
            r'^\d+$',  # Just numbers
            r'^page\s+\d+',
            r'^\d+\s*$',
            r'^p\.\s*\d+',
            r'^第\s*\d+\s*页',  # Chinese page numbers
            r'^ページ\s*\d+',   # Japanese page numbers

            # Common metadata
            r'^figure\s+\d+',
            r'^table\s+\d+',
            r'^fig\.\s*\d+',
            r'^tab\.\s*\d+',
            r'^图\s*\d+',      # Chinese figure
            r'^表\s*\d+',      # Chinese table
            r'^図\s*\d+',      # Japanese figure

            # URLs and emails
            r'http[s]?://',
            r'\S+@\S+\.\S+',

            # File paths and technical strings
            r'^[a-zA-Z]:\\',
            r'^/[a-zA-Z0-9/]+',

            # Common footer/header text
            r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # Dates
            r'^copyright\s',
            r'^©\s',
            r'^\d+\s+(of|/)\s+\d+',  # Page x of y

            # Very short strings that are likely noise (adjusted for CJK)
            r'^[a-zA-Z]{1,2}$',  # Single/double letters only
            r'^[^a-zA-Z\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]*$',  # No letters/CJK at all
        ]

        # Enhanced non-heading starters (multilingual)
        self.non_heading_starters = {
            # English
            'the', 'this', 'that', 'these', 'those', 'a', 'an', 'and', 'or', 'but',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'about',
            'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'among', 'up', 'down', 'out', 'off', 'over', 'under',
            'it', 'he', 'she', 'they', 'we', 'you', 'i', 'me', 'him', 'her',
            'them', 'us', 'my', 'your', 'his', 'hers', 'its', 'our', 'their',
            'as', 'if', 'when', 'where', 'why', 'how', 'what', 'which', 'who',
            'whom', 'whose', 'while', 'although', 'though', 'because', 'since',
            'unless', 'until', 'so', 'yet', 'however', 'therefore', 'thus',
            'moreover', 'furthermore', 'nevertheless', 'nonetheless',

            # Chinese common starters that aren't headings
            '这', '那', '这些', '那些', '一个', '一些', '很多', '许多',
            '在', '从', '到', '为', '和', '或', '但', '如果', '当',

            # Japanese common starters
            'この', 'その', 'あの', 'これら', 'それら', 'あれら',
            'で', 'に', 'を', 'が', 'は', 'と', 'や', 'も', 'から',
        }

    def normalize_text(self, text: str) -> str:
        """Enhanced text normalization with better CJK support"""
        if not text:
            return ""

        # Unicode normalization - use NFKC for better CJK handling
        text = unicodedata.normalize('NFKC', text)

        # Remove excessive whitespace but preserve single spaces
        text = re.sub(r'\s+', ' ', text)

        # More careful punctuation removal - preserve CJK punctuation
        # Only remove leading/trailing ASCII punctuation
        text = re.sub(r'^[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\?/`~]*', '', text)
        text = re.sub(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\?/`~]*$', '', text)

        return text.strip()

    def is_cjk_text(self, text: str) -> bool:
        """Check if text contains CJK characters"""
        cjk_pattern = r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]'
        return bool(re.search(cjk_pattern, text))

    def is_excluded_text(self, text: str) -> bool:
        """Enhanced exclusion check with CJK support"""
        text_lower = text.lower().strip()

        # Check exclusion patterns
        for pattern in self.exclusion_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True

        # Check if starts with non-heading words (skip for CJK text)
        if not self.is_cjk_text(text):
            first_word = text_lower.split()[0] if text_lower.split() else ""
            if first_word in self.non_heading_starters:
                return True

        # Exclude if it's mostly punctuation or numbers (adjusted for CJK)
        # Count CJK characters as alphanumeric
        cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', text))
        alphanumeric_chars = sum(1 for c in text if c.isalnum()) + cjk_chars

        if len(text) > 0 and alphanumeric_chars / len(text) < 0.4:
            return True

        # Exclude very repetitive text
        unique_chars = set(text.replace(' ', ''))
        if len(unique_chars) <= 2 and len(text) > 5:
            return True

        return False

    def extract_text_blocks(self, doc: fitz.Document) -> List[TextBlock]:
        """Extract text blocks with enhanced UTF-8 handling"""
        text_blocks = []

        for page_num in range(len(doc)):
            try:
                page = doc.load_page(page_num)
                blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)

                for block in blocks.get("blocks", []):
                    if "lines" not in block:
                        continue

                    for line in block["lines"]:
                        line_text = ""
                        total_size = 0
                        total_flags = 0
                        fonts = []
                        span_count = 0
                        line_bbox = None

                        for span in line["spans"]:
                            # Ensure proper UTF-8 handling
                            text = span.get("text", "")
                            if isinstance(text, bytes):
                                text = text.decode('utf-8', errors='ignore')
                            text = text.strip()

                            if text:
                                line_text += text + " "
                                total_size += span.get("size", 0)
                                total_flags |= span.get("flags", 0)
                                fonts.append(span.get("font", ""))
                                span_count += 1

                                if line_bbox is None:
                                    line_bbox = span.get("bbox", (0, 0, 0, 0))

                        if line_text.strip() and span_count > 0:
                            normalized_text = self.normalize_text(line_text)

                            # Skip if text is excluded
                            if self.is_excluded_text(normalized_text):
                                continue

                            # Skip if text is too short (but allow shorter CJK text)
                            min_len = 1 if self.is_cjk_text(normalized_text) else self.min_heading_length
                            if len(normalized_text) < min_len:
                                continue

                            avg_size = total_size / span_count
                            dominant_font = max(set(fonts), key=fonts.count) if fonts else ""

                            text_block = TextBlock(
                                text=normalized_text,
                                size=avg_size,
                                flags=total_flags,
                                font=dominant_font,
                                page=page_num + 1,
                                bbox=line_bbox or (0, 0, 0, 0),
                                is_bold=bool(total_flags & (1 << 4)),
                                is_italic=bool(total_flags & (1 << 6))
                            )

                            text_blocks.append(text_block)

            except Exception as e:
                logger.warning(f"Error processing page {page_num + 1}: {str(e)}")
                continue

        return text_blocks

    def is_likely_heading(self, block: TextBlock, avg_size: float, max_size: float) -> bool:
        """Enhanced heading detection with better CJK support"""
        text = block.text.strip()

        # Adjusted length constraints for CJK
        min_len = 1 if self.is_cjk_text(text) else self.min_heading_length
        max_len = self.max_heading_length * 2 if self.is_cjk_text(text) else self.max_heading_length

        if len(text) < min_len or len(text) > max_len:
            return False

        # Skip if excluded
        if self.is_excluded_text(text):
            return False

        # Skip sentences (adjusted for CJK - different punctuation)
        if not self.is_cjk_text(text):
            if text.endswith('.') and len(text) > 30 and text.count('.') == 1:
                return False
        else:
            # For CJK, check for sentence-ending punctuation
            if len(text) > 20 and text.endswith(('。', '．', '!')):
                return False

        # Skip text with too many commas (adjusted for CJK)
        comma_chars = [',', '，', '、'] if self.is_cjk_text(text) else [',']
        comma_count = sum(text.count(c) for c in comma_chars)
        if comma_count > 2:
            return False

        # Check for heading patterns
        has_pattern = any(re.match(pattern, text, re.IGNORECASE | re.UNICODE)
                          for pattern in self.heading_patterns)

        # Check for heading keywords
        text_lower = text.lower()
        has_keyword = any(
            keyword in text_lower
            for keyword in self.heading_keywords
        )

        # Font size analysis
        size_factor = block.size / avg_size if avg_size > 0 else 1
        is_significantly_larger = size_factor >= self.min_heading_size_ratio

        # Bold text check
        is_bold = block.is_bold

        # Position check
        is_left_aligned = block.bbox[0] < 150

        # Enhanced capitalization check for multilingual
        has_title_case = False
        if not self.is_cjk_text(text):
            words = text.split()
            if words:
                capitalized_words = sum(1 for word in words if word and word[0].isupper())
                has_title_case = capitalized_words / len(words) >= 0.5
        else:
            # For CJK, consider it as having title case
            has_title_case = True

        # All caps check (not applicable to CJK)
        is_all_caps = not self.is_cjk_text(text) and text.isupper() and len(text) <= 50

        # Scoring system with CJK adjustments
        score = 0
        if has_pattern: score += 4
        if has_keyword: score += 3
        if is_significantly_larger: score += 3
        if is_bold: score += 2
        if is_left_aligned: score += 1
        if has_title_case: score += 1
        if is_all_caps: score += 2

        # Very large fonts are likely headings
        if block.size >= max_size * 0.85:
            score += 3

        # Lower threshold for CJK text as patterns might be different
        threshold = 3 if self.is_cjk_text(text) else 4
        return score >= threshold

    def determine_heading_level(self, block: TextBlock, all_headings: List[TextBlock]) -> str:
        """Enhanced heading level determination with CJK support"""
        text = block.text.strip()

        # Enhanced pattern-based level detection
        h1_patterns = [
            r'^(Chapter|CHAPTER|章|第.*章)\s*\d*',
            r'^(Part|PART|部|第.*部)\s*[IVX\d]*',
            r'^\d+\s+[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ffA-Z\s]+$',
            r'^[IVX]+\.\s+[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ffA-Z]',
            r'^第[一二三四五六七八九十百千万\d]+[章部編]\s*',
        ]

        h2_patterns = [
            r'^\d+\.\d+\s+',
            r'^[A-Z]\.\s+[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ffA-Z]',
            r'^\d+\.\s+[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ffA-Z]',
            r'^第[一二三四五六七八九十百千万\d]+[节節項]\s*',
        ]

        h3_patterns = [
            r'^\d+\.\d+\.\d+\s+',
            r'^[a-z]\)\s+',
            r'^\(\d+\)\s+',
            r'^[a-z]\.\s+',
            r'^[一二三四五六七八九十]\s*[、．]\s*',
        ]

        # Check patterns first
        for pattern in h1_patterns:
            if re.match(pattern, text, re.UNICODE):
                return "H1"

        for pattern in h2_patterns:
            if re.match(pattern, text, re.UNICODE):
                return "H2"

        for pattern in h3_patterns:
            if re.match(pattern, text, re.UNICODE):
                return "H3"

        # Size-based classification
        if not all_headings:
            return "H1"

        sizes = [h.size for h in all_headings]
        unique_sizes = sorted(list(set(sizes)), reverse=True)

        if len(unique_sizes) <= 2:
            return "H1" if block.size >= max(sizes) * 0.95 else "H2"
        else:
            if block.size >= unique_sizes[0] * 0.95:
                return "H1"
            elif len(unique_sizes) >= 2 and block.size >= unique_sizes[1] * 0.95:
                return "H2"
            else:
                return "H3"

    def extract_title(self, text_blocks: List[TextBlock]) -> str:
        """Enhanced title extraction with multilingual support"""
        if not text_blocks:
            return "Untitled Document"

        # Look in first 3 pages for title
        early_blocks = [b for b in text_blocks if b.page <= 3]

        if not early_blocks:
            return "Untitled Document"

        title_candidates = []

        for block in early_blocks:
            # More lenient length check for CJK
            min_len = 2 if self.is_cjk_text(block.text) else 5
            max_len = 150 if self.is_cjk_text(block.text) else 100

            if len(block.text) < min_len or len(block.text) > max_len:
                continue

            if self.is_excluded_text(block.text):
                continue

            # Skip obvious body text (adjusted for CJK)
            if not self.is_cjk_text(block.text):
                if block.text.endswith('.') and len(block.text) > 50:
                    continue
            else:
                if len(block.text) > 30 and block.text.endswith(('。', '．')):
                    continue

            title_candidates.append(block)

        if not title_candidates:
            for block in early_blocks:
                if len(block.text.strip()) >= 2:
                    return block.text
            return "Untitled Document"

        # Find largest font size among candidates
        max_size = max(b.size for b in title_candidates)
        largest_candidates = [b for b in title_candidates if b.size >= max_size * 0.95]

        # Enhanced candidate scoring
        best_candidate = None
        best_score = 0

        for candidate in largest_candidates:
            score = 0
            text = candidate.text

            # Title case check (adjusted for CJK)
            if not self.is_cjk_text(text):
                words = text.split()
                if words and sum(1 for w in words if w and w[0].isupper()) / len(words) > 0.5:
                    score += 2
            else:
                score += 2  # CJK titles are considered properly formatted

            # Position preference
            if candidate.bbox[0] > 50:
                score += 1

            # Length preference (adjusted for CJK)
            ideal_min = 3 if self.is_cjk_text(text) else 10
            ideal_max = 30 if self.is_cjk_text(text) else 60
            if ideal_min <= len(text) <= ideal_max:
                score += 1

            # Avoid numbered headings
            if not any(re.match(p, text, re.UNICODE) for p in self.heading_patterns):
                score += 1

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate.text if best_candidate else largest_candidates[0].text

    def remove_duplicates(self, headings: List[Dict]) -> List[Dict]:
        """Enhanced duplicate removal with better CJK handling"""
        if not headings:
            return []

        unique_headings = []
        seen_texts = {}

        for heading in headings:
            text = heading['text']

            # Create normalized key for comparison
            if self.is_cjk_text(text):
                # For CJK, be more strict about exact matches
                normalized = re.sub(r'\s+', '', text.strip())
            else:
                normalized = re.sub(r'\s+', ' ', text.lower().strip())
                normalized = re.sub(r'[^\w\s]', '', normalized)

            # Check for duplicates
            is_duplicate = False
            for seen_key, seen_heading in seen_texts.items():
                if seen_key == normalized:
                    is_duplicate = True
                    break

                # For non-CJK text, check similarity
                if not self.is_cjk_text(text) and len(normalized) > 5 and len(seen_key) > 5:
                    similarity = len(set(normalized.split()) & set(seen_key.split())) / len(set(normalized.split()) | set(seen_key.split()))
                    if similarity > 0.8 and abs(seen_heading['page'] - heading['page']) <= 2:
                        is_duplicate = True
                        break

            if not is_duplicate:
                seen_texts[normalized] = heading
                unique_headings.append(heading)

        return unique_headings

    def process_pdf(self, pdf_path: str) -> Dict:
        """Process PDF with enhanced multilingual support"""
        start_time = time.time()
        doc = None

        try:
            logger.info(f"Processing PDF: {pdf_path}")

            # Open PDF with enhanced text extraction
            doc = fitz.open(pdf_path)

            if len(doc) == 0:
                logger.warning(f"Empty PDF: {pdf_path}")
                return {"title": "Empty Document", "outline": []}

            # Extract text blocks
            text_blocks = self.extract_text_blocks(doc)

            if not text_blocks:
                logger.warning(f"No text found in PDF: {pdf_path}")
                return {"title": "No Text Found", "outline": []}

            # Calculate size statistics
            sizes = [b.size for b in text_blocks if b.size > 0]
            if not sizes:
                return {"title": "No Valid Text", "outline": []}

            avg_size = sum(sizes) / len(sizes)
            max_size = max(sizes)

            # Extract title
            title = self.extract_title(text_blocks)

            # Filter potential headings
            potential_headings = [
                block for block in text_blocks
                if self.is_likely_heading(block, avg_size, max_size)
            ]

            # Remove title from headings if it appears
            title_normalized = self.normalize_text(title.lower())
            potential_headings = [
                h for h in potential_headings
                if self.normalize_text(h.text.lower()) != title_normalized
            ]

            # Determine heading levels
            outline = []
            for block in potential_headings:
                level = self.determine_heading_level(block, potential_headings)
                outline.append({
                    "level": level,
                    "text": block.text,  # Keep original text with proper encoding
                    "page": block.page
                })

            # Remove duplicates and sort
            outline = self.remove_duplicates(outline)
            outline.sort(key=lambda x: (x["page"], x["level"]))

            processing_time = time.time() - start_time
            logger.info(f"Processed {pdf_path}: {len(outline)} headings in {processing_time:.2f}s")

            return {
                "title": title,
                "outline": outline
            }

        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return {
                "title": "Error Processing Document",
                "outline": []
            }
        finally:
            if doc is not None:
                try:
                    doc.close()
                except:
                    pass

def validate_output(result: Dict) -> bool:
    """Enhanced validation with better Unicode support"""
    if not isinstance(result, dict):
        return False

    if "title" not in result or "outline" not in result:
        return False

    if not isinstance(result["title"], str):
        return False

    if not isinstance(result["outline"], list):
        return False

    for item in result["outline"]:
        if not isinstance(item, dict):
            return False

        required_keys = {"level", "text", "page"}
        if not all(key in item for key in required_keys):
            return False

        if item["level"] not in ["H1", "H2", "H3"]:
            return False

        if not isinstance(item["text"], str) or not item["text"].strip():
            return False

        if not isinstance(item["page"], int) or item["page"] < 1:
            return False

    return True

def main():
    """Main execution function with enhanced UTF-8 support"""
    # Set UTF-8 encoding for stdout
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    # Determine directories based on environment
    if os.path.exists("/app/input"):
        input_dir = Path("/app/input")
        output_dir = Path("/app/output")
    else:
        input_dir = Path("./input")
        output_dir = Path("./output")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        print(f"ERROR: Input directory '{input_dir}' not found")
        sys.exit(1)

    # Find PDF files
    pdf_files = list(input_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found")
        print(f"No PDF files found in {input_dir}")
        return

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    # Initialize extractor
    extractor = PDFOutlineExtractor()

    # Process files
    successful = 0
    failed = 0

    for pdf_file in pdf_files:
        try:
            result = extractor.process_pdf(str(pdf_file))

            # Validate output
            if not validate_output(result):
                logger.error(f"Invalid output format for {pdf_file}")
                failed += 1
                continue

            # Save result with proper UTF-8 encoding and ensure_ascii=False
            output_file = output_dir / f"{pdf_file.stem}.json"

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, separators=(',', ': '))

            # Display results with proper encoding
            heading_count = len(result['outline'])
            title_preview = result['title'][:50] + ('...' if len(result['title']) > 50 else '')

            print(f"✓ {pdf_file.name} → {output_file.name}")
            print(f"  Title: {title_preview}")
            print(f"  Headings: {heading_count}")

            # Show sample headings with proper encoding
            if result['outline']:
                print("  Sample headings:")
                for i, heading in enumerate(result['outline'][:3]):
                    text_preview = heading['text'][:40] + ('...' if len(heading['text']) > 40 else '')
                    print(f"    {heading['level']}: {text_preview} (p.{heading['page']})")
                if len(result['outline']) > 3:
                    print(f"    ... and {len(result['outline']) - 3} more")
            print()

            successful += 1

        except UnicodeEncodeError as e:
            logger.error(f"Unicode encoding error for {pdf_file}: {str(e)}")
            print(f"✗ Unicode Error: {pdf_file.name}")
            failed += 1
        except Exception as e:
            logger.error(f"Failed to process {pdf_file}: {str(e)}")
            print(f"✗ Failed: {pdf_file.name} - {str(e)}")
            failed += 1

    # Summary
    print(f"Processing complete: {successful} successful, {failed} failed")

    if successful > 0:
        print(f"JSON files saved to: {output_dir}")
        print("Note: Chinese/Japanese text is preserved in original encoding")

if __name__ == "__main__":
    main()