import chromadb
from PIL import Image

def initialize_chroma_db(path, name):
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(name=name)
    return collection

def add_embeddings(collection, ids, embeddings, metadatas):
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas
    )

def query_embeddings(collection, query_embedding, n_results=5):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results


def load_image_from_result(metadata):
    """Given a single result metadata dict, load the actual image."""
    return Image.open(metadata['image_path'])


if __name__ == "__main__":
    from args import parse_args
    args = parse_args()
    collection = initialize_chroma_db(args)

    print("ChromaDB and CLIP model initialized successfully")
