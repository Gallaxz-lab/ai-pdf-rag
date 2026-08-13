from pypdf import PdfReader

def extract_pages_from_pdf(pdf_path: str) -> list[dict]:
    """
    Reads a local PDF file path page-by-page.
    Returns a list of dictionaries tracking text and literal page numbers.
    """
    try:
        reader = PdfReader(pdf_path)
        pages_data = []
        for index, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages_data.append({
                    "page_number": index + 1,  # Human-readable 1-indexed count
                    "text_content": text
                })
        return pages_data
    except Exception as e:
        raise ValueError(f"Failed reading PDF layout: {str(e)}")

def split_page_data_into_chunks(pages_data: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    """
    Slices tracked page data using character sliding windows while retaining page metadata.
    Returns a dictionary list containing chunk texts and accurate parent page sources.
    """
    structured_chunks = []
    chunk_global_counter = 0

    for page in pages_data:
        text = page["text_content"]
        page_num = page["page_number"]
        start_index = 0
        text_length = len(text)

        while start_index < text_length:
            end_index = start_index + chunk_size
            chunk_slice = text[start_index:end_index].strip()
            
            if chunk_slice:
                structured_chunks.append({
                    "chunk_id": f"chunk_sz{chunk_size}_{chunk_global_counter}",
                    "text": chunk_slice,
                    "metadata": {
                        "page": page_num,
                        "chunk_index": chunk_global_counter,
                        "chunk_size_mode": chunk_size
                    }
                })
                chunk_global_counter += 1
            
            start_index += (chunk_size - chunk_overlap)
            
    return structured_chunks
