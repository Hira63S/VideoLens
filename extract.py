from src.encoder.clip_encoder import get_clip_model, encode_text
from src.vector_db.chroma import initialize_chroma_db, query_embeddings, load_image_from_result
from src.args import parse_args
import os


args = parse_args()

model, tokenizer, preprocess = get_clip_model(args)
collection = initialize_chroma_db(args.chroma_path, args.db_name)

query = query = "nighttime scenes"
query_embedding = encode_text(model, tokenizer, query)

results = query_embeddings(collection, query_embedding, n_results=args.n_results)

for i, metadata in enumerate(results['metadatas'][0]):
    print(f"Scene: {metadata['scene']}, Frame: {metadata['frame_idx']}, Path: {metadata['image_path']}")
    img = load_image_from_result(metadata)
    os.makedirs("query_results", exist_ok=True)
    img.save(f"query_results/result_{i}.png")