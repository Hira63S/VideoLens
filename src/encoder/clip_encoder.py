import open_clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_clip_model(args):

    model, _, preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained='openai')
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    model = model.to(device)
    return model, tokenizer, preprocess

def encode_images(model, preprocess, images):

    # do we prepare the image inputs here? 
    image_pils = [Image.fromarray(img) for img in images]
    image_inputs = torch.stack([preprocess(img) for img in image_pils]).to(device)
    with torch.no_grad():
        clip_embeddings = model.encode_image(image_inputs)
        clip_embeddings = clip_embeddings / clip_embeddings.norm(dim=-1, keepdim=True)
    return clip_embeddings.cpu().numpy().tolist()

def encode_text(model, tokenizer, text):
    text_embedding = tokenizer([text]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text_embedding)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
    return text_embedding.cpu().numpy().tolist()[0]

if __name__ == "__main__":
    from args import parse_args
    args = parse_args()
    model, tokenizer, preprocess = get_clip_model(args)
    print("CLIP model initialized successfully")