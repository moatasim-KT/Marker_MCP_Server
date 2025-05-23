from marker.providers.pdf import PdfProvider
import tempfile

import datasets


def setup_pdf_provider(
    filename='adversarial.pdf',
    config=None,
) -> PdfProvider:
    dataset = datasets.load_dataset("datalab-to/pdfs", split="train")

    pdf_bytes_content = None
    # Iterate through the dataset to find the item with the matching filename
    for item in dataset:
        item_filename = item.get("filename")
        if item_filename == filename:
            pdf_bytes_content = item.get("pdf")  # 'pdf' field in this dataset contains bytes
            break

    if pdf_bytes_content is None:
        raise ValueError(f"File '{filename}' not found in the dataset.")

    temp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf")
    temp_pdf.write(pdf_bytes_content) # This now correctly writes bytes
    temp_pdf.flush()

    provider = PdfProvider(temp_pdf.name, config)
    return provider
