from src.encoder.clip_encoder import get_clip_model, encode_text
from src.vector_db.chroma import initialize_chroma_db, query_embeddings, load_image_from_result
from src.dataloader.nuscenes_dataset import NuScenesLoader
from src.args import parse_args
import os
import cv2



def search_videos(query):
    args = parse_args()

    model, tokenizer, preprocess = get_clip_model(args)
    collection = initialize_chroma_db(args.chroma_path, args.db_name)
    query_embedding = encode_text(model, tokenizer, query)

    results = query_embeddings(collection, query_embedding, n_results=args.n_results)

    sorted_results = sorted(
        zip(results['ids'][0], results['metadatas'][0]),
        key=lambda x: x[1]["frame_idx"]
    )

    return sorted_results
'''
{'ids': [['scene-0061_7', 'scene-0061_4', 'scene-0061_2', 'scene-0061_6', 'scene-0061_5']], 
'embeddings': None, 'documents': [[None, None, None, None, None]], 'uris': None, 
'included': ['metadatas', 'documents', 'distances'], 
'data': None, 'metadatas': [[{'scene': 'scene-0061', 'timestamp': 1532402931198511}, 
{'scene': 'scene-0061', 'timestamp': 1532402929697797}, 
{'scene': 'scene-0061', 'timestamp': 1532402928698048}, 
{'timestamp': 1532402930648325, 'scene': 'scene-0061'}, 
{'timestamp': 1532402930152601, 'scene': 'scene-0061'}]], 
'distances': [[1.5682153701782227, 1.5742344856262207, 1.5761489868164062, 1.5770388841629028, 1.5786325931549072]]}
'''

# these are the outputs we are getting from the database
# now if i wanted to retrieve the actual frames, i would need to use the metadata to find the corresponding frames in the dataset and load them for further processing (e.g., object detection, tracking, etc.)
# the metadata contains the scene name and timestamp, which can be used to locate the specific frames in the dataset. 
# You would typically have a function that takes the scene name and timestamp, queries the dataset, and retrieves the corresponding frame for further analysis.
'''
def postprocess(results):

    # get a scene from the resutls:
    # this will give us the names of the scenes
    scene_list = results['ids'][0]
   ;''' 