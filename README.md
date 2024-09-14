
# Deepfake Detector

This project is a deep learning-based system for detecting deepfakes by analyzing video, audio, and physiological features (PPG signals). The system uses multimodal transformers, LSTMs, and GANs for supervised and unsupervised learning, allowing it to analyze videos for authenticity.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Model Architecture](#model-architecture)
5. [Dataset](#dataset)
6. [Training](#training)
7. [Testing](#testing)
8. [Contributing](#contributing)
9. [License](#license)

## Overview

The deepfake detector is designed to identify fake videos by leveraging both supervised and unsupervised learning techniques. It combines multimodal features, such as video frames and audio data, with physiological analysis of PPG (photoplethysmography) signals to improve accuracy. The model can process videos from local storage or YouTube.

### Key Features:
- **Multimodal Input**: The system processes both video and audio data, integrating them using transformers.
- **PPG Signal Analysis**: Extracts PPG signals from video frames for physiological analysis.
- **Deep Learning Models**: Uses transformers for multimodal integration, LSTMs for sequence modeling, and GANs for unsupervised anomaly detection.
- **KnowledgeMap**: A custom SQLite database that stores extracted features for future reference.

## Installation

### Prerequisites

Make sure Python 3.x is installed along with the following dependencies:

```bash
pip install torch torchvision transformers pytube google-api-python-client moviepy statsmodels opencv-python librosa scikit-learn
git clone https://github.com/forestreal/deepfakedetector.git
cd deepfakedetector


YouTube API Key
This project uses the YouTube API to download videos for training and testing. Get your YouTube API key from here and update the key in the code:

python
Copy code
API_KEY = "YOUR_YOUTUBE_API_KEY"
Usage

After setting up the dependencies and YouTube API key, you can run the deepfake detector.

Running the Detector
bash
Copy code
python deepfakedetector.py
When prompted, provide a video path or YouTube video ID to test the model.

Example:

bash
Copy code
Please provide the path to your video:
The model will process the input video and display whether it is classified as a real or deepfake.

Model Architecture

1. Multimodal Transformer
The core of the system integrates both audio and video features using a transformer. Video and audio features are projected into a shared latent space for comparison.
2. LSTM for Sequence Modeling
PPG signals extracted from video frames are processed with an LSTM to capture temporal dependencies, improving robustness in detecting subtle inconsistencies in fake videos.
3. GAN for Unsupervised Learning
The GAN (Generative Adversarial Network) helps identify deepfakes in an unsupervised manner by modeling real and fake distributions. The discriminator is trained to detect anomalous patterns in videos that may indicate forgery.
Dataset

Sources
FaceForensics++: Pre-processed deepfake video dataset.
YouTube: Custom videos are fetched using the YouTube API, allowing dynamic testing with celebrity videos or user-submitted videos.
KnowledgeMap
The KnowledgeMap is an SQLite database used to store video features (AR and PPG) for reference in future tests.

Training

Supervised Learning
The MultimodalTransformer model is trained with labeled datasets containing both real and fake videos. The transformer integrates audio and video data for a more holistic analysis.

To train the model:

bash
Copy code
python train_supervised.py
Unsupervised Learning (GAN)
The GAN model learns to detect anomalous patterns through unsupervised learning. The generator creates synthetic data while the discriminator distinguishes between real and generated data.

To train the GAN:

bash
Copy code
python train_unsupervised_gan.py
Both models are fine-tuned using a dataset of real and deepfake videos, leveraging dynamic feature extraction from YouTube.

Testing

After training the model, you can run the following command to test any video:

bash
Copy code
python deepfakedetector.py
You will be prompted to provide a path to a local video or a YouTube video ID. The model will output whether the video is classified as "Real" or "Deepfake".

Example Output:
bash
Copy code
Deepfake Detection Result: Deepfake
Possible source videos:
https://www.youtube.com/watch?v=exampleVideoID1
https://www.youtube.com/watch?v=exampleVideoID2
Contributing

Contributions are welcome! Follow these steps to contribute:

Fork the repository.
Create a new branch.
Commit your changes.
Push to your branch.
Open a Pull Request.
License

This project is licensed under the MIT License.

