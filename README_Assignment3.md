## Assignment 3: MNIST GAN

This extension adds a PyTorch GAN to the existing SPS GenAI FastAPI project, without changing the existing embedding and CIFAR-10 classifier endpoints.

### Architecture

- Generator: noise (B, 100) -> Linear to 128 x 7 x 7 -> ConvTranspose2d 128 -> 64 (14 x 14) -> BatchNorm2d -> ReLU -> ConvTranspose2d 64 -> 1 (28 x 28) -> Tanh.
- Discriminator: grayscale image (B, 1, 28, 28) -> Conv2d 1 -> 64 (14 x 14) -> LeakyReLU(0.2) -> Conv2d 64 -> 128 (7 x 7) -> BatchNorm2d -> LeakyReLU(0.2) -> Flatten -> Linear -> Sigmoid.
- Loss: binary cross-entropy. Optimization: Adam, learning rate 0.0002, betas (0.5, 0.999).
- Data: MNIST, pixel values normalized to [-1, 1] to match the generator's Tanh output.

### Train and test

```powershell
uv run python gan_model.py
uv run python train_gan.py --epochs 3
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/docs> and try **GET /generate**, with count=16. The endpoint returns a PNG grid of generated digits. You can also open <http://127.0.0.1:8000/generate?count=16> directly in a browser.

The training script saves `mnist_gan_generator.pth` after every epoch, as well as PNG samples in `gan_samples/` and loss records in `gan_training_history.json`. Because GAN training is stochastic, outputs will improve with additional training, but may still contain imperfect digits.

For a short end-to-end smoke test (not a substitute for full MNIST training):

```powershell
uv run python train_gan.py --epochs 1 --max-batches 10
```

The trained weights are needed for `/generate`. If no checkpoint is present, the endpoint returns HTTP 503 with instructions instead of returning untrained noise.
