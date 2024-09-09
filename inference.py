import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Compose, Resize
from PIL import Image
import argparse
import os
import json
import cv2  # OpenCV library for video processing
from models import AdvancedCNN  # Import if you're using a custom model
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from datasets import AnimalDataset

def get_args():
    parser = argparse.ArgumentParser(description="Inference using a trained model")
    parser.add_argument("--input-path", "-i", type=str, required=True, help="Path to the image or video")
    parser.add_argument("--checkpoint-path", "-c", type=str, default="trained_models/best.pt", help="Path to the model checkpoint")
    parser.add_argument("--image-size", "-s", type=int, default=224, help="Size of image")
    parser.add_argument("--output-path", "-o", type=str, default="output.json", help="Path to save the predictions")
    parser.add_argument("--video-output-path", "-vo", type=str, help="Path to save the output video with predictions (optional)")
    parser.add_argument("--frame-skip", type=int, default=1, help="Number of frames to skip during video processing")
    args = parser.parse_args()
    return args

def load_model(checkpoint_path, num_classes):
    # Load the model architecture
    model = mobilenet_v2(weights=MobileNet_V2_Weights)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)

    # Load the model weights from the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.eval()  # Set the model to evaluation mode
    return model

def process_image(image_path, transform):
    # Open the image file
    image = Image.open(image_path)
    # Apply transformations
    image = transform(image)
    # Add batch dimension
    image = image.unsqueeze(0)
    return image

def process_video_frame(frame, transform):
    pass

def predict(model, image, device):
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

def process_image_input(model, args, transform, device, class_names):
    image = process_image(args.input_path, transform)
    predicted_class = predict(model, image, device)
    prediction = {args.input_path: class_names[predicted_class]}

    # Save the prediction to a JSON file
    with open(args.output_path, 'w') as f:
        json.dump(prediction, f)
    print(f"Prediction saved to {args.output_path}")

def process_video_input(model, args, transform, device, class_names):
    pass

def main():
    args = get_args()
    
    # Load the dataset to get class names
    dataset = AnimalDataset(root="data/animals", is_train=False, transform=None)
    class_names = dataset.categories

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(args.checkpoint_path, num_classes=len(class_names))
    model.to(device)

    # Define transform
    transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
    ])

    # Check if input is an image or video
    if args.input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        process_image_input(model, args, transform, device, class_names)
    elif args.input_path.lower().endswith(('.mp4', '.avi', '.mov')):
        process_video_input(model, args, transform, device, class_names)
    else:
        print("Unsupported file format. Please provide an image or video file.")

if __name__ == "__main__":
    main()
