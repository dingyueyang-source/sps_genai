# Assignment 4 — CIFAR-10 Image Generation

Course repository [dingyueyang-source/sps_genai](https://github.com/dingyueyang-source/sps_genai).

The Assignment 4 router is registered on the existing FastAPI app, where the earlier embedding, CIFAR-10 classifier, and MNIST GAN routes run. The new endpoints load their included CIFAR-10 checkpoints on CPU and return RGB PNG grids. The CIFAR-10 dataset is only needed for training.

## Docker

From the Canvas submission directory:

```sh
docker build -t sps-genai-a4 .
docker run --rm -p 8000:8000 sps-genai-a4
```

In the GitHub checkout, build the cumulative API with:

```sh
docker build --target integrated -f Dockerfile.assignment4 -t sps-genai-a4 .
docker run --rm -p 8000:8000 sps-genai-a4
```

The standalone kit also has a default Docker build that runs just the two new endpoints. Use `docker build -t sps-genai-a4 .` from its extracted directory.

Open [interactive API documentation](http://localhost:8000/docs) or try:

- [Diffusion PNG](http://localhost:8000/generate/diffusion?count=4&steps=50&seed=42)
- [Energy model PNG](http://localhost:8000/generate/energy?count=4&steps=300&seed=42)

Both endpoints accept `count`, `steps`, and an optional `seed`. The response's `X-Training-Steps` header reports the checkpoint's number of training updates. Sample responses are in `assignment4/examples/`.

## Training

Train from the project root in an environment with PyTorch and torchvision. Training uses the CIFAR-10 training split, 32 × 32 RGB images and pixel values scaled to [−1, 1]. The default seed is 42 and the batch size is 32.

```sh
python -m assignment4.train diffusion --steps 3500 --threads 5
python -m assignment4.train energy --steps 1500 --threads 3
```

The diffusion model uses a time-conditioned U-Net, cosine noise schedule and a mean squared error loss against the sampled noise. It generates images from Gaussian noise with DDIM updates. The energy model outputs one scalar for each image. In training, Langevin updates generate negative samples from a replay buffer with occasional random restarts. The loss lowers energy on training images and raises it on generated ones, with a squared-energy regularizer. Langevin sampling differentiates energy with respect to the image and updates pixels while the model weights stay fixed.

The final diffusion logging window had a mean noise MSE of 0.07240. The samples show color and coarse structure, with limited object detail. Training logs are included in `assignment4/examples/`. These are compact CPU-trained checkpoints.

## Written answers

`Assignment4_Answers.pdf` covers questions 1–9. The examples for questions 6–9 can be run with `python -m assignment4.answers_autograd`.
