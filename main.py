"""SPS GenAI API: word embeddings, CIFAR-10 classification, and MNIST GAN."""

import io
from pathlib import Path
from functools import lru_cache

import spacy
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.utils import make_grid

from cnn_model import SimpleCNN
from gan_model import Generator, LATENT_DIM

app = FastAPI(title="SPS GenAI API")
nlp = spacy.load("en_core_web_lg")

root = Path(__file__).resolve().parent
torch.set_num_threads(4)

classifier = SimpleCNN()
classifier.load_state_dict(
    torch.load(root / "cifar10_cnn.pth", map_location="cpu", weights_only=True)
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
    return {"word": word, "embedding": vector.tolist(), "dimensions": len(vector)}


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        with Image.open(io.BytesIO(contents)) as image:
            inputs = image_transform(image.convert("RGB")).unsqueeze(0)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Please upload a valid image.")
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


@lru_cache(maxsize=1)
def load_gan_generator():
    """Load the trained model once, after train_gan.py has saved it."""
    weights = root / "mnist_gan_generator.pth"
    if not weights.is_file():
        raise HTTPException(
            status_code=503,
            detail="GAN weights not found. Run: uv run python train_gan.py --epochs 3",
        )
    model = Generator()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()
    return model


@app.get("/generate", responses={200: {"content": {"image/png": {}}}})
def generate_digits(count: int = Query(default=16, ge=1, le=64)):
    """Return a PNG grid of handwritten digits generated from random noise."""
    generator = load_gan_generator()
    with torch.no_grad():
        noise = torch.randn(count, LATENT_DIM)
        images = generator(noise)
        grid = make_grid(images, nrow=min(8, count), normalize=True,
                         value_range=(-1, 1), padding=2)
    pil_image = transforms.ToPILImage()(grid)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")
