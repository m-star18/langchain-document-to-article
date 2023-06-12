import argparse

from langchain.document_loaders import OnlinePDFLoader, UnstructuredMarkdownLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, NLTKTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings


class MakeDocumentEmbedding:
    """MakeDocumentEmbedding _summary_

    _extended_summary_
    """
    def __init__(self, args) -> None:
        self.open_api = args.OPEN_API_KEY
        self.chunk_size = args.chunk_size
        self.chunk_overlap = args.chunk_overlap
        self.file_path = args.file_path
        self.split_mode = args.split_mode
        self.embedding = OpenAIEmbeddings(openai_api_key=self.open_api)

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

    def text_splitter(self):
        """text_splitter _summary_

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
        # Split by separator and merge by character count
        if self.split_mode == "character":
            # Create a CharacterTextSplitter object
            text_splitter = CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        # Recursively split until below the chunk size limit
        elif self.split_mode == "recursive_character":
            # Create a RecursiveCharacterTextSplitter object
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        elif self.split_mode == "nltk":
            # Create a NLTKTextSplitter object
            text_splitter = NLTKTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        elif self.split_mode == "tiktoken":
            # Create a CharacterTextSplitter object
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        else:
            raise ValueError("Please specify the split mode.")

        return text_splitter.split_documents(self.document_data)

    def make_embeddings(self):
        """make_embeddings _summary_

        _extended_summary_

        Returns
        -------
        _type_
            _description_
        """

        # Create a FAISS object
        faiss_db = FAISS.from_documents(self.document_data, self.embedding)

        return faiss_db

    def save_embeddings(self, save_path):
        """save_embeddings _summary_

        _extended_summary_

        Parameters
        ----------
        save_path : _type_
            _description_
        """
        # Save the FAISS object
        self.faiss_db.save_local(save_path)


def make_emb(args):
    make_document_embedding = MakeDocumentEmbedding(args)
    if args.faiss_save_path:
        make_document_embedding.save_embeddings(args.faiss_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--OPEN_API_KEY", type=str, default="")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Please specify the chunk_size for CharacterTextSplitter within a number less than or equal to 4096.")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Please specify the chunk_overlap for CharacterTextSplitter within a number less than or equal to 4096.")
    parser.add_argument("--file_path", type=str, default="", help="Please specify the path of the article to be read. (pdf, docx, md)")
    parser.add_argument("--split_mode", type=str, default="recursive_character", help="Please specify the split mode. (character, recursive_character, nltk, tiktoken)")
    parser.add_argument("--faiss_save_path", type=str, default="faiss_index", help="Please specify the name of the created Faiss object.")
    args = parser.parse_args()

    make_emb(args)
