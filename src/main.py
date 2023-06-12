import argparse

from src.make_embedding import *


def config_parser():
    """_summary_

    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--OPEN_API_KEY", type=str, default="")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Please specify the chunk_size for CharacterTextSplitter within a number less than or equal to 4096.")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Please specify the chunk_overlap for CharacterTextSplitter within a number less than or equal to 4096.")
    parser.add_argument("--file_path", type=str, default="", help="Please specify the path of the article to be read. (pdf, docx, md)")
    parser.add_argument("--split_mode", type=str, default="recursive_character", help="Please specify the split mode. (character, recursive_character, nltk, tiktoken)")
    parser.add_argument("--faiss_save_path", type=str, default="faiss_index", help="Please specify the name of the created Faiss object.")

    return parser


def main():
    parser = config_parser()
    args = parser.parse_args()

    make_emb(args)


if __name__ == "__main__":
    main()
