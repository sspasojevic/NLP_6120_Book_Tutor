import re
import fitz  # PyMuPDF for PDF processing
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

def open_and_read_pdf(pdf_path):
    """
    Opens a PDF file and extracts text from each page.

    Args:
        pdf_path (str): The file path to the PDF document.

    Returns:
        list of dict: A list where each dictionary contains the page number and its corresponding text.
    """
    # Open the PDF document using PyMuPDF
    doc = fitz.open(pdf_path)
    pages_and_texts = []

    # Iterate over each page in the PDF with a progress bar
    for page_num, page in tqdm(enumerate(doc), total=len(doc), desc="Reading PDF pages"):
        # Extract text from the current page
        text = page.get_text()
        # Append the page number and text to the list
        pages_and_texts.append({
            "page_number": page_num + 1,
            "text": text
        })
    return pages_and_texts

def clean_text(text):
    """
    Cleans a given text by removing unnecessary whitespace and line breaks.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    """
    Cleans a paragraph by:
    1. Replacing " \n" with a single space.
    """
    # Replace " \n" with a single space
    text = re.sub(r" \n", " ", text)

    # Remove extra spaces and strip leading/trailing whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def split_into_paragraphs(pages_and_texts):
    """
    Splits the combined text from all PDF pages into paragraphs based on defined delimiters.

    Args:
        pages_and_texts (list of dict): List containing page numbers and their corresponding texts.

    Returns:
        pd.DataFrame: A DataFrame where each row represents a paragraph with metadata.
    """
    # Define the pattern to identify paragraph boundaries
    paragraph_delimiter = r"(?:\s*\n\s*\n\s*|\s{2,}\n)"

    # Combine text from all pages and track page boundaries
    combined_text = ""
    page_boundaries = []
    for page_data in pages_and_texts:
        start_idx = len(combined_text)  # Starting index for the current page
        combined_text += page_data["text"]
        page_boundaries.append((start_idx, len(combined_text), page_data["page_number"]))

    # Split the combined text into paragraphs using the delimiter pattern
    paragraphs = re.split(paragraph_delimiter, combined_text)

    # Initialize a list to hold paragraph data
    paragraph_data = []
    for paragraph in paragraphs:
        # Clean the paragraph text
        cleaned_paragraph = clean_text(paragraph)

        # Skip empty paragraphs after cleaning
        if not cleaned_paragraph:
            continue

        # Skip paragraphs with fewer than 20 words
        if len(paragraph.split(" ")) < 20:
            continue

        # Find the start and end indices of the paragraph in the combined text
        paragraph_start_idx = combined_text.find(paragraph)
        paragraph_end_idx = paragraph_start_idx + len(paragraph)
        pages_spanned = set()

        # Determine which pages the paragraph spans
        for start, end, page_number in page_boundaries:
            if paragraph_start_idx < end and paragraph_end_idx > start:
                pages_spanned.add(page_number)

        # Append the paragraph data to the list
        paragraph_data.append({
            "page_number": sorted(pages_spanned),
            "char_count": len(cleaned_paragraph),
            "word_count": len(cleaned_paragraph.split(" ")),
            "sentence_count": len(sent_tokenize(cleaned_paragraph)),
            "text": cleaned_paragraph
        })

    # Convert the list of dictionaries to a pandas DataFrame
    return pd.DataFrame(paragraph_data)

def split_into_chunks(pages_and_texts, chunk_size):
    """
    Splits the combined text from all PDF pages into fixed-size chunks.

    Args:
        pages_and_texts (list of dict): List containing page numbers and their corresponding texts.
        chunk_size (int): The number of characters per chunk.

    Returns:
        pd.DataFrame: A DataFrame where each row represents a text chunk with metadata.
    """
    # Combine text from all pages and track page boundaries
    combined_text = ""
    page_boundaries = []
    for page_data in pages_and_texts:
        start_idx = len(combined_text)  # Starting index for the current page
        combined_text += page_data["text"]
        page_boundaries.append((start_idx, len(combined_text), page_data["page_number"]))

    # Split the combined text into chunks of specified size
    chunks = [combined_text[i:i + chunk_size] for i in range(0, len(combined_text), chunk_size)]

    # Initialize a list to hold chunk data
    chunk_data = []
    for chunk in chunks:
        # Clean the chunk text
        cleaned_chunk = clean_text(chunk)

        # Skip empty chunks after cleaning
        if not cleaned_chunk:
            continue

        # Find the start and end indices of the chunk in the combined text
        chunk_start_idx = combined_text.find(chunk)
        chunk_end_idx = chunk_start_idx + len(chunk)
        pages_spanned = set()

        # Determine which pages the chunk spans
        for start, end, page_number in page_boundaries:
            if chunk_start_idx < end and chunk_end_idx > start:
                pages_spanned.add(page_number)

        # Append the chunk data to the list
        chunk_data.append({
            "page_number": sorted(pages_spanned),
            "char_count": len(cleaned_chunk),
            "word_count": len(cleaned_chunk.split(" ")),
            "sentence_count": len(sent_tokenize(cleaned_chunk)),
            "text": cleaned_chunk
        })

    # Convert the list of dictionaries to a pandas DataFrame
    return pd.DataFrame(chunk_data)

def split_into_sentences(pages_and_texts, num_sentences=10):
    """
    Splits the combined text from all PDF pages into chunks of a specified number of sentences.

    Args:
        pages_and_texts (list of dict): List containing page numbers and their corresponding texts.
        num_sentences (int, optional): Number of sentences per chunk. Defaults to 10.

    Returns:
        pd.DataFrame: A DataFrame where each row represents a sentence-based chunk with metadata.
    """
    # Combine text from all pages and track page boundaries
    combined_text = ""
    page_boundaries = []
    for page_data in pages_and_texts:
        start_idx = len(combined_text)  # Starting index for the current page
        combined_text += page_data["text"]
        page_boundaries.append((start_idx, len(combined_text), page_data["page_number"]))

    # Define the pattern to identify sentence boundaries
    sentence_boundary_pattern = r'(?<=[.!?])(?=\s|\n)'

    # Split the combined text into individual sentences
    sentences = re.split(sentence_boundary_pattern, combined_text)

    # Group sentences into chunks of the specified number
    chunks = ["".join(sentences[i:i + num_sentences]) for i in range(0, len(sentences), num_sentences)]

    # Initialize a list to hold chunk data
    chunk_data = []
    for chunk in chunks:
        # Clean the chunk text
        cleaned_chunk = clean_text(chunk)

        # Skip empty chunks after cleaning
        if not cleaned_chunk:
            continue

        # Find the start and end indices of the chunk in the combined text
        chunk_start_idx = combined_text.find(chunk)
        chunk_end_idx = chunk_start_idx + len(chunk)
        pages_spanned = set()

        # Determine which pages the chunk spans
        for start, end, page_number in page_boundaries:
            if chunk_start_idx < end and chunk_end_idx > start:
                pages_spanned.add(page_number)

        # Append the chunk data to the list
        chunk_data.append({
            "page_number": sorted(pages_spanned),
            "char_count": len(cleaned_chunk),
            "word_count": len(cleaned_chunk.split(" ")),
            "sentence_count": len(sent_tokenize(cleaned_chunk)),
            "text": cleaned_chunk
        })

    # Convert the list of dictionaries to a pandas DataFrame
    return pd.DataFrame(chunk_data)

def split_into_pages(pages_and_texts):
    """
    Organizes the text from each PDF page into a DataFrame without further splitting.

    Args:
        pages_and_texts (list of dict): List containing page numbers and their corresponding texts.

    Returns:
        pd.DataFrame: A DataFrame where each row represents a page with metadata.
    """
    pages_data = []

    # Iterate over each page's data
    for page_data in pages_and_texts:
        # Clean the page text
        cleaned_page = clean_text(page_data["text"])

        # Append the page data to the list
        pages_data.append({
            "page_number": page_data["page_number"],
            "char_count": len(cleaned_page),
            "word_count": len(cleaned_page.split(" ")),
            "sentence_count": len(sent_tokenize(cleaned_page)),
            "text": cleaned_page
        })

    # Convert the list of dictionaries to a pandas DataFrame
    return pd.DataFrame(pages_data)

def create_df_from_pdf(pdf_path, method="sentence", fixed_size=512, num_sentences=10):
    """
    Creates a pandas DataFrame from a PDF by splitting its text based on the specified method.

    Args:
        pdf_path (str): The file path to the PDF document.
        method (str, optional): The method to split the text ("paragraph", "fixed", "sentence", "page"). Defaults to "sentence".
        fixed_size (int, optional): The number of characters per chunk for the "fixed" method. Defaults to 512.
        num_sentences (int, optional): The number of sentences per chunk for the "sentence" method. Defaults to 10.

    Returns:
        pd.DataFrame: A DataFrame containing the split text with associated metadata.
    """
    # Open and read the PDF to extract text from each page
    pages_and_texts = open_and_read_pdf(pdf_path)

    # Split the text based on the specified method
    if method == "paragraph":
        df = split_into_paragraphs(pages_and_texts)
    elif method == "fixed":
        df = split_into_chunks(pages_and_texts, fixed_size)
    elif method == "sentence":
        df = split_into_sentences(pages_and_texts, num_sentences)
    elif method == "page":
        df = split_into_pages(pages_and_texts)
    else:
        raise ValueError(f"Unsupported method: {method}. Choose from 'paragraph', 'fixed', 'sentence', or 'page'.")

    return df
