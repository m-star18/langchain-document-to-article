import numpy as np
from langchain.vectorstores.faiss import dependable_faiss_import


def merge_faiss_indexes(index1, index2):
    # Extract vectors from both indexes
    vectors1 = np.array(index1.index.reconstruct_n(0, index1.index.ntotal))
    vectors2 = np.array(index2.index.reconstruct_n(0, index2.index.ntotal))

    # Concatenate the vectors.
    concatenated_vectors = np.vstack((vectors1, vectors2))

    # Create a new index.
    faiss = dependable_faiss_import()
    merged_index = faiss.IndexFlatL2(concatenated_vectors.shape[1])

    # Add the concatenated vectors to the merged index.
    merged_index.add(concatenated_vectors)

    # Replace the old index with the merged one.
    index1.index = merged_index

    # Merge docstore and index_to_docstore_id dictionaries (use your own strategy)
    index1.docstore._dict.update(index2.docstore._dict)
    new_length = len(index1.index_to_docstore_id)
    index1.index_to_docstore_id.update({i + new_length: id for i, id in index2.index_to_docstore_id.items()})

    return index1
