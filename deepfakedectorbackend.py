import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from pytube import YouTube
from googleapiclient.discovery import build
import sqlite3
from statsmodels.tsa.ar_model import AutoReg
import cv2
import numpy as np
import os
from moviepy.editor import VideoFileClip
import pickle

# YouTube API Key (replace with your API key)
API_KEY = "YOUR_YOUTUBE_API_KEY"

# Hyperparameters
learning_rate = 0.001
batch_size = 4
epochs = 10  # Supervised learning epochs
gan_epochs = 5  # GAN training epochs
hidden_dim = 512  # LSTM/Transformer hidden dimension
n_heads = 8  # Transformer attention heads
num_layers = 4  # Transformer layers

# SQLite KnowledgeMap for storing AR/PPG features
class KnowledgeMap:
    def __init__(self, db_path='knowledge_map.db'):
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        self._create_table()

    def _create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS features (
                video_id TEXT PRIMARY KEY,
                ar_features BLOB,
                ppg_features BLOB
            )
        ''')
        self.connection.commit()

    def store(self, video_id, features):
        ar_features_blob = pickle.dumps(features['ar'])  # Serialize AR features
        ppg_features_blob = pickle.dumps(features['ppg'])  # Serialize PPG features
        self.cursor.execute('''
            INSERT OR REPLACE INTO features (video_id, ar_features, ppg_features)
            VALUES (?, ?, ?)
        ''', (video_id, ar_features_blob, ppg_features_blob))
        self.connection.commit()

    def retrieve(self, video_id):
        self.cursor.execute('SELECT ar_features, ppg_features FROM features WHERE video_id = ?', (video_id,))
        row = self.cursor.fetchone()
        if row:
            ar_features = pickle.loads(row[0])  # Deserialize AR features
            ppg_features = pickle.loads(row[1])  # Deserialize PPG features
            return {'ar': ar_features, 'ppg': ppg_features}
        return None

    def close(self):
        self.connection.close()

# Function to download a video from YouTube using the video_id and return the file path
def get_video_path(video_id):
    try:
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        yt = YouTube(youtube_url)
        stream = yt.streams.filter(file_extension="mp4").first()

        download_dir = "videos"  # Folder to store videos
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        video_path = stream.download(download_dir)

        return video_path
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None

# Enhanced PPG extraction from video frames
def extract_ppg_from_video(frames):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    ppg_signals = []
    
    for frame in frames:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_roi = frame[y:y+h, x:x+w]
            green_channel = face_roi[:, :, 1]
            ppg_signal = np.mean(green_channel)
            ppg_signals.append(ppg_signal)

    return np.array(ppg_signals)

# AR feature extraction using AutoRegressive model from raw video
def extract_ar_features_from_video(frames, lag=2):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    intensity_signals = []

    for frame in frames:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_roi = frame[y:y+h, x:x+w]
            intensity_signal = np.mean(face_roi[:, :, 1])
            intensity_signals.append(intensity_signal)

    intensity_signals = np.array(intensity_signals)

    if len(intensity_signals) > lag:
        model = AutoReg(intensity_signals, lags=lag)
        ar_fit = model.fit()
        return ar_fit.params
    else:
        return np.zeros(lag)

# Preprocessing function
def preprocess_and_extract_features(video_path):
    video_input = dynamic_frame_selection(video_path)
    ar_features = extract_ar_features_from_video(video_input)
    ppg_signal = extract_ppg_from_video(video_input)
    audio_input = extract_audio_with_moviepy(video_path)
    return video_input, audio_input, ar_features, ppg_signal

# Function for dynamic frame selection
def dynamic_frame_selection(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames)

# Function to extract audio from video using moviepy
def extract_audio_with_moviepy(video_path):
    clip = VideoFileClip(video_path)
    audio = clip.audio
    audio_array = audio.to_soundarray()
    clip.close()
    return audio_array

# YouTube search function using googleapiclient
def search_youtube(query, max_results=5):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    request = youtube.search().list(
        part="snippet",
        q=query,
        maxResults=max_results
    )
    response = request.execute()
    video_ids = [item['id']['videoId'] for item in response['items'] if 'videoId' in item['id']]
    return video_ids

# Multimodal Transformer Model with LSTM for supervised learning
class MultimodalTransformer(nn.Module):
    def __init__(self, video_dim, audio_dim, hidden_dim, n_heads, num_layers):
        super(MultimodalTransformer, self).__init__()
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, num_layers=2)
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=n_heads, num_encoder_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, video_features, audio_features, ar_features, ppg_signal):
        video_proj = self.video_proj(video_features)
        audio_proj = self.audio_proj(audio_features)
        
        multimodal_features = torch.cat([video_proj.unsqueeze(1), audio_proj.unsqueeze(1)], dim=1)
        transformer_out = self.transformer(multimodal_features, multimodal_features)

        ar_ppg_combined = torch.stack([ar_features, ppg_signal], dim=-1)
        lstm_out, _ = self.lstm(ar_ppg_combined.unsqueeze(0))
        
        combined_features = transformer_out.mean(dim=0) + lstm_out[:, -1, :]
        output = self.fc(combined_features)
        return self.sigmoid(output)

# GAN for unsupervised learning
class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 224*224),
            nn.Tanh()
        )
        self.discriminator = nn.Sequential(
            nn.Linear(224*224, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)

# Training supervised model
def train_supervised(model, video_ids):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()
    
    for epoch in range(epochs):
        for video_id in video_ids:
            video_path = get_video_path(video_id)
            if video_path:
                video_features, audio_features, ar_features, ppg_signal = preprocess_and_extract_features(video_path)
                optimizer.zero_grad()
                output = model(video_features, audio_features, ar_features, ppg_signal)
                target = torch.tensor([1.0])
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
            else:
                print(f"Failed to process video with ID: {video_id}")
        print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item()}")

# Training GAN model
def train_unsupervised_gan(gan_model, video_ids):
    optimizer = torch.optim.Adam(gan_model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()
    
    for epoch in range(gan_epochs):
        for video_id in video_ids:
            video_path = get_video_path(video_id)
            if video_path:
                video_features = preprocess_and_extract_features(video_path)[0]
                optimizer.zero_grad()
                
                # GAN training steps
                real_data = video_features.view(-1, 224*224)
                fake_data = gan_model.generator(torch.randn(batch_size, 100)).detach()
                
                # Discriminator step
                real_output = gan_model.discriminator(real_data)
                fake_output = gan_model.discriminator(fake_data)
                
                real_loss = loss_fn(real_output, torch.ones(batch_size, 1))
                fake_loss = loss_fn(fake_output, torch.zeros(batch_size, 1))
                loss = real_loss + fake_loss
                
                loss.backward()
                optimizer.step()
            else:
                print(f"Failed to process video with ID: {video_id}")
        
        print(f"Epoch {epoch+1}/{gan_epochs} GAN Loss: {loss.item()}")

# User input and final report
def user_input_and_report():
    user_video_path = input("Please provide the path to your video: ")
    
    # Process user video
    user_video_input, user_audio_input, user_ar_features, user_ppg_signal = preprocess_and_extract_features(user_video_path)
    
    # Query YouTube for similar videos
    video_ids = search_youtube("celebrity face video", max_results=5)
    
    # Initialize models
    transformer_model = MultimodalTransformer(video_dim=2048, audio_dim=768, hidden_dim=hidden_dim, n_heads=n_heads, num_layers=num_layers)
    gan_model = GAN()

    # Train models
    train_supervised(transformer_model, video_ids)
    train_unsupervised_gan(gan_model, video_ids)

    # Test user video on the trained model
    output = transformer_model(user_video_input, user_audio_input, user_ar_features, user_ppg_signal)
    result = "Real" if output.item() > 0.5 else "Deepfake"

    print(f"Deepfake Detection Result: {result}")
    
    if result == "Deepfake":
        print("Possible source videos:")
        for vid_id in video_ids:
            print(f"https://www.youtube.com/watch?v={vid_id}")

# Call the user input function
if __name__ == "__main__":
    user_input_and_report()




