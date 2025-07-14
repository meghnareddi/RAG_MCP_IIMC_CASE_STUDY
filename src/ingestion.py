from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.schema import Document
from collections import defaultdict
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from config import embedding_function
import re

import re
from collections import defaultdict

def clean_pdf_elements(elements, position_margin_ratio=0.1, repetition_threshold=0.8):
    """
    Remove headers and footers by:
    - Tracking occurrences of (text, y_bucket) per page
    - Remove elements whose (text, y_bucket) appears on >= threshold fraction of pages
    - Remove elements in top/bottom margin of page height
    """

    occurrence_tracker = defaultdict(set)
    element_positions = {}

    # Extract all page numbers and max page for total pages
    all_pages = set()
    for el in elements:
        page = el.metadata.get("page_number", -1)
        if page >= 0:
            all_pages.add(page)
    total_pages = max(all_pages) + 1 if all_pages else 1

    for el in elements:
        text = " ".join(el.page_content.strip().split())
        page = el.metadata.get("page_number", -1)
        coords = el.metadata.get("coordinates", {})
        y_coords = [pt[1] for pt in coords.get("points", [])]

        # Use layout_height from metadata or fallback to 841 (A4)
        page_height = el.metadata.get("layout_height", 841)

        mean_y = sum(y_coords) / len(y_coords) if y_coords else None
        rounded_y = round(mean_y / 20) * 20 if mean_y else None

        key = (text, rounded_y)
        if page >= 0:
            occurrence_tracker[key].add(page)

        element_positions[el.metadata.get("element_id")] = key

    # Identify keys appearing on >= repetition_threshold fraction of pages
    repeated_keys = {
        key for key, pages in occurrence_tracker.items()
        if len(pages) / total_pages >= repetition_threshold
    }

    cleaned_elements = []
    seen_texts = set()

    for el in elements:
        element_id = el.metadata.get("element_id")
        key = element_positions.get(element_id)
        text = key[0] if key else el.page_content.strip()
        coords = el.metadata.get("coordinates", {})
        y_coords = [pt[1] for pt in coords.get("points", [])]

        # Get page height again for margin check
        page_height = el.metadata.get("layout_height", 841)
        top_margin = page_height * position_margin_ratio
        bottom_margin = page_height * (1 - position_margin_ratio)

        # Skip if repeated header/footer by threshold
        if key in repeated_keys:
            continue

        # Skip elements fully in top or bottom margin area
        if y_coords:
            y_min, y_max = min(y_coords), max(y_coords)
            if y_max < top_margin or y_min > bottom_margin:
                continue

        # Skip short or duplicate texts
        if len(text) < 50 or text in seen_texts:
            continue
        seen_texts.add(text)

        cleaned_elements.append(el)

    print(f"✅ Retained {len(cleaned_elements)} / {len(elements)} elements after cleaning")
    return cleaned_elements

def load_documents(path):
    document_loader = UnstructuredPDFLoader(path, mode="elements")
    elements = document_loader.load()
    cleaned_elements = clean_pdf_elements(elements)
    return cleaned_elements



def chunk_documents(documents, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

def create_embeddings_store(documents, persist_directory="db"):
    filtered_documents = filter_complex_metadata(documents)
    vectordb = Chroma.from_documents(
        filtered_documents,
        embedding=embedding_function,
        persist_directory=persist_directory
    )
    vectordb.persist()  
    #print("Embeddings created and stored successfully.")
    return vectordb

""" def remove_headers_footers(text):
    header_patterns = [
        r'^The decade ahead:.*$',
        r'^Trends that will shape the consumer goods industry.*$'
    ]
    footer_patterns = [
        r'^\s*\d+\s*$',          # page numbers
        r'^\d+\s.*$'             # footnote references
    ]

    for pattern in header_patterns + footer_patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE)

    return text.strip() """

