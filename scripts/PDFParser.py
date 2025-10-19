import pytesseract

from typing import Any
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf

class PDFParser:
    """
    
    """
    def __init__(self, path):
        self.path = path
    def parse(self):
        """
        """
        chunks = partition_pdf(
            filename=self.path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=4000,
            combine_text_under_n_chars=1000,
            new_after_n_chars=3000
        )
        tables_list = []
        texts_list = []
        for chunk in chunks:
            if "Table" in str(type(chunk)):
                tables_list.append(chunk)
            if "CompositeElement" in str(type(chunk)):
                texts_list.append(chunk)
        images_list = self.get_images_base64(chunks)
        return texts_list, tables_list, images_list
    
    def _get_images_base64(self, chunks):
        images_b64 = []
        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
                chunk_els = chunk.metadata.orig_elements
                for el in chunk_els:
                    if "Image" in str(type(el)):
                        images_b64.append(el.metadata.image_base64)
        return images_b64