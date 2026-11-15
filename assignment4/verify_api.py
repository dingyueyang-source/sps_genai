"""Verify the generators and save example PNG responses."""

import argparse
from importlib import import_module
from io import BytesIO
from pathlib import Path

import torch
from fastapi.testclient import TestClient
from PIL import Image


def verify(module, output):
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=True)
    client = TestClient(import_module(module).app)
    if module == "main":
        paths = client.get("/openapi.json").json()["paths"]
        for path in ("/embedding", "/classify", "/generate"):
            if path not in paths:
                raise AssertionError(f"Missing prior route {path}")

    for model in ("diffusion", "energy"):
        url = f"/generate/{model}"
        params = {"count": 8, "seed": 42, "steps": 50 if model == "diffusion" else 300}
        response = client.get(url, params=params)
        response.raise_for_status()
        assert response.headers["content-type"] == "image/png"
        assert int(response.headers["X-Training-Steps"]) > 0
        with Image.open(BytesIO(response.content)) as image:
            assert image.format == "PNG" and image.mode == "RGB" and image.size == (138, 70)
            image.verify()
        assert client.get(url, params=params).content == response.content
        assert client.get(url, params={**params, "count": 0}).status_code == 422
        (output / f"{model}.png").write_bytes(response.content)
        print(f"GET {url} -> 200 PNG, {response.headers['X-Training-Steps']} training steps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", default="assignment4.service")
    parser.add_argument("--output", type=Path, default=Path("assignment4/examples"))
    args = parser.parse_args()
    verify(args.module, args.output)
