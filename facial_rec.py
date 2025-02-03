import os
import cv2
import torch
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# ==== Configurable Variables ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
DEFAULT_MODEL_PATH = "face_model.pth"  # File to save/load face embeddings
DEFAULT_CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for face detection
DEFAULT_SIMILARITY_THRESHOLD = 0.6  # Cosine similarity threshold for recognition
DEFAULT_IMAGE_FOLDER = "images"  # Folder for training images
# ================================

# Load a pre-trained Faster R-CNN model for face detection
print("Loading face detection model...")
face_detector = fasterrcnn_resnet50_fpn(weights="DEFAULT")
face_detector.eval().to(DEVICE)


def extract_face_embeddings(image, face_boxes):
    """
    Extract face embeddings using pixel-based features (for simplicity).
    Replace with a proper embedding model for better accuracy.
    """
    embeddings = []
    for box in face_boxes:
        x1, y1, x2, y2 = map(int, box)
        face = image[:, y1:y2, x1:x2]
        resized_face = F.resize(face, (128, 128))  # Resize to a fixed size
        flattened = resized_face.flatten().float() / 255.0  # Normalize
        embeddings.append(flattened)
    return embeddings


def train(image_folder, model_path, confidence_threshold):
    """
    Train a model using labeled images in the specified folder.
    """
    embeddings = []
    labels = []

    print(f"Training from images in: {image_folder}")
    for file_name in os.listdir(image_folder):
        file_path = os.path.join(image_folder, file_name)

        # Ensure it's an image
        if not os.path.isfile(file_path) or not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # Prompt for label
        label = input(f"Enter label for {file_name} (or 'skip'): ").strip()
        if label.lower() == "skip":
            continue

        # Detect faces in the image
        image = cv2.imread(file_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor_image = F.to_tensor(rgb_image).unsqueeze(0).to(DEVICE)
        detections = face_detector(tensor_image)[0]

        # Filter detections by threshold
        face_boxes = [
            box.cpu().detach().numpy()
            for box, score in zip(detections["boxes"], detections["scores"])
            if score > confidence_threshold
        ]

        # Extract embeddings
        face_embeddings = extract_face_embeddings(rgb_image, face_boxes)
        embeddings.extend(face_embeddings)
        labels.extend([label] * len(face_embeddings))

    # Save embeddings and labels
    torch.save({"embeddings": torch.stack(embeddings), "labels": labels}, model_path)
    print(f"Model saved to {model_path}")


def recognize(model_path, similarity_threshold):
    """
    Perform real-time face recognition using the webcam.
    """
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Train the model first.")
        return

    # Load embeddings and labels
    data = torch.load(model_path)
    stored_embeddings = data["embeddings"]
    stored_labels = data["labels"]

    cap = cv2.VideoCapture(0)
    print("Starting webcam. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Convert frame to RGB and detect faces
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor_frame = F.to_tensor(rgb_frame).unsqueeze(0).to(DEVICE)
        detections = face_detector(tensor_frame)[0]

        # Filter detections by confidence threshold
        face_boxes = [
            box.cpu().detach().numpy()
            for box, score in zip(detections["boxes"], detections["scores"])
            if score > DEFAULT_CONFIDENCE_THRESHOLD
        ]

        for box in face_boxes:
            x1, y1, x2, y2 = map(int, box)
            face_embedding = extract_face_embeddings(rgb_frame, [box])[0]

            # Compare with stored embeddings
            similarities = torch.nn.functional.cosine_similarity(
                stored_embeddings, face_embedding.unsqueeze(0), dim=-1
            )
            max_similarity, best_match_idx = torch.max(similarities, dim=0)

            # Determine label
            label = stored_labels[best_match_idx] if max_similarity > similarity_threshold else "Unknown"

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({max_similarity:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display frame
        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PyTorch-based Face Detection and Recognition",
        epilog="Examples:\n  python facial_rec.py train --image-folder images/ --model-path model.pth\n  python facial_rec.py recognize --model-path model.pth --similarity-threshold 0.7\n\nYou can also use train or recognize --help for more options.\n",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Training arguments
    train_parser = subparsers.add_parser("train", help="Train a face recognition model")
    train_parser.add_argument(
        "--image-folder",
        type=str,
        default=DEFAULT_IMAGE_FOLDER,
        help="Path to the folder containing labeled training images (default: images/)",
    )
    train_parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to save the trained model (default: face_model.pth)",
    )
    train_parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Confidence threshold for face detection (default: 0.5)",
    )

    # Recognition arguments
    recognize_parser = subparsers.add_parser("recognize", help="Run face recognition in real time")
    recognize_parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to load the trained model (default: face_model.pth)",
    )
    recognize_parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="Cosine similarity threshold for face matching (default: 0.6)",
    )

    args = parser.parse_args()

    if args.command == "train":
        train(args.image_folder, args.model_path, args.confidence_threshold)
    elif args.command == "recognize":
        recognize(args.model_path, args.similarity_threshold)