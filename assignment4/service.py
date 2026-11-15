from functools import lru_cache
from io import BytesIO
from pathlib import Path

import torch
from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.responses import Response
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from .models import Diffusion, EnergyModel, langevin, uniform_images

router = APIRouter(tags=["CIFAR-10 Image Generation"])
CHECKPOINTS = Path(__file__).with_name("checkpoints")
PNG_RESPONSE = {200: {"content": {"image/png": {}}}}


@lru_cache(maxsize=2)
def load_model(name):
    path = CHECKPOINTS / f"{name}.pt"
    if not path.is_file():
        raise HTTPException(503, f"Missing checkpoint: {name}.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = Diffusion() if name == "diffusion" else EnergyModel()
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval().requires_grad_(False), checkpoint["train_steps"]


def request_rng(seed):
    rng = torch.Generator()
    if seed is None:
        rng.seed()
    else:
        rng.manual_seed(seed)
    return rng


def png_grid(images, trained_steps):
    grid = make_grid(
        images, nrow=min(4, len(images)), normalize=True, value_range=(-1, 1)
    )
    buffer = BytesIO()
    to_pil_image(grid).save(buffer, format="PNG")
    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={"X-Training-Steps": str(trained_steps)},
    )


@router.get("/generate/diffusion", response_class=Response, responses=PNG_RESPONSE)
def generate_diffusion(
    count: int = Query(4, ge=1, le=16),
    steps: int = Query(50, ge=2, le=100),
    seed: int | None = Query(None, ge=0, le=2**32 - 1),
):
    """Generate a PNG grid with the CIFAR-10 diffusion model."""
    model, trained_steps = load_model("diffusion")
    return png_grid(model.sample(count, steps, request_rng(seed)), trained_steps)


@router.get("/generate/energy", response_class=Response, responses=PNG_RESPONSE)
def generate_energy(
    count: int = Query(4, ge=1, le=16),
    steps: int = Query(200, ge=1, le=500),
    seed: int | None = Query(None, ge=0, le=2**32 - 1),
):
    """Generate a PNG grid by applying Langevin updates to random images."""
    model, trained_steps = load_model("energy")
    rng = request_rng(seed)
    initial = uniform_images(count, "cpu", rng)
    return png_grid(langevin(model, initial, steps, rng=rng), trained_steps)


# Provides a lightweight entry point for running the two new routes on their own.
app = FastAPI(title="SPS GenAI API | Assignment 4")
app.include_router(router)
