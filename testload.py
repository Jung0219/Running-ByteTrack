import torch

file_path = '/home/finn/ByteTrack/best.pt'
try:
    model = torch.load(file_path, map_location='cpu')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load the model: {e}")
