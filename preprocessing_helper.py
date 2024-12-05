import re
import fitz # PyMuPDF
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

def open_and_read_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages_and_texts = []

    for page_num, page in tqdm(enumerate(doc), total=len(doc)):
        text = page.get_text()
        pages_and_texts.append({
            "page_number": page_num + 1,
            "text": text
        })
    return pages_and_texts

def clean_text(text):
    """
    Cleans a paragraph by:
    1. Replacing " \\n" with a single space.
    """

    # Replace " \n" with a single space
    text = re.sub(r" \n", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def split_into_paragraphs(pages_and_texts):

    paragraph_delimiter = r"(?:\s*\n\s*\n\s*|\s{2,}\n)"

    # Combine text from all pages
    combined_text = ""
    page_boundaries = []
    for page_data in pages_and_texts:
        start_idx = len(combined_text)  # Track starting index for this page
        combined_text += page_data["text"]
        page_boundaries.append((start_idx, len(combined_text), page_data["page_number"]))

    # Split combined text into paragraphs
    paragraphs = re.split(paragraph_delimiter, combined_text)

    # Map paragraphs back to pages
    paragraph_data = []
    for paragraph in paragraphs:

        cleaned_paragraph = clean_text(paragraph)

        if not cleaned_paragraph:  # Skip empty paragraphs after cleaning
            continue

        if len(paragraph.split(" ")) < 20:
            continue

        # Find the pages the paragraph corresponds to
        paragraph_start_idx = combined_text.find(paragraph)
        paragraph_end_idx = paragraph_start_idx + len(paragraph)
        pages_spanned = set()

        for start, end, page_number in page_boundaries:
            if paragraph_start_idx < end and paragraph_end_idx > start:
                pages_spanned.add(page_number)

        paragraph_data.append({
            "page_number": sorted(pages_spanned),
            "char_count": len(cleaned_paragraph),
            "word_count": len(cleaned_paragraph.split(" ")),
            "sentence_count": len(sent_tokenize(cleaned_paragraph)),
            "text": cleaned_paragraph
        })

    return pd.DataFrame(paragraph_data)

def split_into_chunks(pages_and_texts, chunk_size):

    # Combine text from all pages
    combined_text = ""
    page_boundaries = []
    for page_data in pages_and_texts:
        start_idx = len(combined_text)  # Track starting index for this page
        combined_text += page_data["text"]
        page_boundaries.append((start_idx, len(combined_text), page_data["page_number"]))

    chunks = [combined_text[i:i + chunk_size] for i in range(0, len(combined_text), chunk_size)]

    chunk_data = []
    for chunk in chunks:

        cleaned_chunk = clean_text(chunk)

        if not cleaned_chunk:  # Skip empty paragraphs after cleaning
            continue

        # Find the pages the paragraph corresponds to
        paragraph_start_idx = combined_text.find(chunk)
        paragraph_end_idx = paragraph_start_idx + len(chunk)
        pages_spanned = set()

        for start, end, page_number in page_boundaries:
            if paragraph_start_idx < end and paragraph_end_idx > start:
                pages_spanned.add(page_number)

        chunk_data.append({
            "page_number": sorted(pages_spanned),
            "char_count": len(cleaned_chunk),
            "word_count": len(cleaned_chunk.split(" ")),
            "sentence_count": len(sent_tokenize(cleaned_chunk)),
            "text": cleaned_chunk
        })

    return pd.DataFrame(chunk_data)

def split_into_sentences(pages_and_texts, num_sentences = 10):

    combined_text = ""
    page_boundaries = []
    for page_data in pages_and_texts:
        start_idx = len(combined_text)  # Track starting index for this page
        combined_text += page_data["text"]
        page_boundaries.append((start_idx, len(combined_text), page_data["page_number"]))

    sentence_boundary_pattern = r'(?<=[.!?])(?=\s|\n)'

    # Split text into sentences
    sentences = re.split(sentence_boundary_pattern, combined_text)

    # Group sentences into chunks of `num_sentences`
    chunks = ["".join(sentences[i:i + num_sentences]) for i in range(0, len(sentences), num_sentences)]

    chunk_data = []
    for chunk in chunks:

        cleaned_chunk = clean_text(chunk)

        if not cleaned_chunk:  # Skip empty chunks after cleaning
            continue

        # Find the pages the chunk corresponds to
        chunk_start_idx = combined_text.find(chunk)
        chunk_end_idx = chunk_start_idx + len(chunk)
        pages_spanned = set()

        for start, end, page_number in page_boundaries:
            if chunk_start_idx < end and chunk_end_idx > start:
                pages_spanned.add(page_number)

        chunk_data.append({
            "page_number": sorted(pages_spanned),
            "char_count": len(cleaned_chunk),
            "word_count": len(cleaned_chunk.split(" ")),
            "sentence_count": len(sent_tokenize(cleaned_chunk)),
            "text": cleaned_chunk
        })

    return pd.DataFrame(chunk_data)

def split_into_pages(pages_and_texts):

    pages_data = []

    for page_data in pages_and_texts:

        cleaned_page = clean_text(page_data["text"])

        pages_data.append({
            "page_number": page_data["page_number"],
            "char_count": len(cleaned_page),
            "word_count": len(cleaned_page.split(" ")),
            "sentence_count": len(sent_tokenize(cleaned_page)),
            "text": cleaned_page
        })

    return pd.DataFrame(pages_data)

def create_df_from_pdf(pdf_path, method="sentence", fixed_size = 512, num_sentences = 10):

    pages_and_texts = open_and_read_pdf(pdf_path)

    if method == "paragraph":
        df = split_into_paragraphs(pages_and_texts)

    if method == "fixed":
        df = split_into_chunks(pages_and_texts, fixed_size)

    if method =="sentence":
        df = split_into_sentences(pages_and_texts, num_sentences)

    if method == "page":
        df = split_into_pages(pages_and_texts)

    return df