import io
from pathlib import Path

import spacy
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from cnn_model import SimpleCNN


app = FastAPI(title="SPS GenAI API")
nlp = spacy.load("en_core_web_lg")

root = Path(__file__).resolve().parent
torch.set_num_threads(4)

classifier = SimpleCNN()
classifier.load_state_dict(
    torch.load(
        root / "cifar10_cnn.pth",
        map_location="cpu",
        weights_only=True,
    )
)
classifier.eval()

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


@app.get("/")
def read_root():
    return {"message": "Hello from SPS GenAI"}


def calculate_embedding(input_word: str):
    return nlp(input_word).vector


@app.get("/embedding")
def get_embedding(word: str):
    vector = calculate_embedding(word)
    return {
        "word": word,
        "embedding": vector.tolist(),
        "dimensions": len(vector),
    }


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        with Image.open(io.BytesIO(contents)) as image:
            inputs = image_transform(
                image.convert("RGB")
            ).unsqueeze(0)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image.",
        )

    with torch.no_grad():
        outputs = classifier(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(dim=1)

    class_id = predicted.item()
    return {
        "filename": file.filename,
        "class_id": class_id,
        "class": classes[class_id],
        "confidence": confidence.item(),
    }
