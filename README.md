# SPS GenAI API

This project includes a spaCy word embedding API and a CIFAR-10 image classifier.

## Setup and run

Python 3.12 and uv are required. Run these commands in the project directory:

    uv sync --locked
    uv run uvicorn main:app --host 127.0.0.1 --port 8000

Open http://127.0.0.1:8000/docs to test the endpoints.

## Endpoints

GET / returns a welcome message.

GET /embedding?word=cat returns a word embedding and its dimensions.

POST /classify accepts an image uploaded in the file field. It returns the filename, class ID, predicted class, and confidence.

The classifier converts images to RGB and resizes them to 64 x 64 before prediction.

## CNN architecture

Input: 3 x 64 x 64.
Conv2d: 16 filters, 3 x 3 kernel, stride 1, padding 1.
ReLU and 2 x 2 max pooling with stride 2.
Conv2d: 32 filters, 3 x 3 kernel, stride 1, padding 1.
ReLU and 2 x 2 max pooling with stride 2.
Flatten: 8192 features.
Fully connected: 100 units, followed by ReLU.
Fully connected: 10 output classes.

Check the architecture with:

    uv run python cnn_model.py

## Training and results

The model was trained on all 50,000 CIFAR-10 training images for 10 epochs, using batch size 32, cross-entropy loss, and Adam with learning rate 0.0005.

Final training accuracy: 76.82%.
Accuracy on the 10,000 test images: 64.97%.

cifar10_cnn.pth contains the trained weights.
training_results.json contains the training history and test accuracy.

The included test_image.png has the true label cat. The API predicted cat with confidence approximately 0.9328 and returned HTTP 200.

To train again, run:

    uv run python train_cnn.py

The dataset is downloaded automatically into data/.
The API uses the included weights, so retraining is not required to run it.
