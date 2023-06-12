import argparse

from langchain.document_loaders import OnlinePDFLoader, UnstructuredMarkdownLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, NLTKTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings


class MakeDocumentEmbedding:
    """ _summary_

    _extended_summary_
    """
    def __init__(self, args) -> None:
        self.open_api = args.OPEN_API_KEY
        self.chunk_size = args.chunk_size
        self.chunk_overlap = args.chunk_overlap
        self.file_path = args.file_path
        self.split_mode = args.split_mode
        self.embedding = OpenAIEmbeddings()

        self.document_data = self.load_document()
        self.split_document_data = self.text_splitter()
        self.faiss_db = self.make_embeddings()
    
    def load_document(self):
        """load_document _summary_

        _extended_summary_

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        # Load the document
        if self.file_path.endswith(".pdf"):
            # Load the PDF file (if the file is a URL, load the PDF file from the URL)
            if self.file_path.startswith("http"):
                loader = OnlinePDFLoader(self.file_path)
            else:
                loader = PyPDFLoader(self.file_path)
        # Load the Word file
        elif self.file_path.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(self.file_path)
        # Load the Markdown file
        elif self.file_path.endswith(".md"):
            loader = UnstructuredMarkdownLoader(self.file_path)
        else:
            raise ValueError("Please specify the path of the PDF file to be read.")
        document_data = loader.load()
        return document_data
