from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str) -> str:
    """Reads a local PDF file path and outputs a unified clean text string."""
    try:
        reader = PdfReader(pdf_path)
        extracted_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text.append(text)
        return "\n".join(extracted_text).strip()
    except Exception as e:
        raise ValueError(f"Failed reading PDF file layers: {str(e)}")

def split_text_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 80) -> list[str]:
    """Slices raw strings into manageable paragraphs using character sliding windows."""
    chunks = []
    start_index = 0
    text_length = len(text)

    while start_index < text_length:
        end_index = start_index + chunk_size
        chunks.append(text[start_index:end_index].strip())
        start_index += (chunk_size - chunk_overlap)
        
    return [c for c in chunks if c]
