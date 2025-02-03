import os
import cv2
import pickle
import argparse
from deepface import DeepFace
from deepface.basemodels import Facenet
from deepface.commons import functions
import numpy as np

# ==== Configurable Variables ====
DEFAULT_MODEL_NAME = "default_model.pkl"  
DEFAULT_WEBCAM_INDEX = 0  
DEFAULT_SHOW_BOXES = True  
DEFAULT_DETECTION_SCALE = 0.5  # Scale factor for face detection (smaller = faster)
DEFAULT_IMAGE_FOLDER = "images"  
DEFAULT_THRESHOLD = 10  # Threshold for face matching
MODEL_NAME = "Facenet"  
# ================================

# Load the DeepFace model
print("Loading DeepFace model...")
model = DeepFace.build_model(MODEL_NAME)


def train_classifier(image_folder: str, model_name: str):
    """
    Train a classifier using images in the specified folder.
    Saves the embeddings and labels to a file.
    """
    embeddings = []
    labels = []

    print(f"Scanning folder: {image_folder}")
    for file_name in os.listdir(image_folder):
        file_path = os.path.join(image_folder, file_name)

        if not os.path.isfile(file_path) or not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # Display the image and prompt for a label
        image = cv2.imread(file_path)
        cv2.imshow("Label This Image", image)
        label = input(f"Enter a label for {file_name} (or 'skip' to ignore): ").strip()

        # Skip unlabeled images
        if label.lower() == "skip":
            continue

        # Detect face and extract embedding
        try:
            embedding = DeepFace.represent(file_path, model_name=MODEL_NAME, model=model)[0]["embedding"]
            embeddings.append(embedding)
            labels.append(label)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue

        print(f"Labeled: {file_name} as {label}")

        cv2.destroyAllWindows()

    with open(model_name, "wb") as f:
        pickle.dump({"embeddings": embeddings, "labels": labels}, f)
    print(f"Model saved to {model_name}")


def recognize_faces(model_name: str, webcam_index: int, show_boxes: bool, detection_scale: float, threshold: float):
 
    if not os.path.exists(model_name):
        print(f"Model file '{model_name}' not found. Train the model first.")
        return

    with open(model_name, "rb") as f:
        data = pickle.load(f)
    embeddings = data["embeddings"]
    labels = data["labels"]

    cap = cv2.VideoCapture(webcam_index)
    print("Starting webcam for real-time recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        # Resize frame for faster detection
        small_frame = cv2.resize(frame, (0, 0), fx=detection_scale, fy=detection_scale)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        faces = functions.detect_faces(rgb_frame, enforce_detection=False)

        for face in faces:
            x, y, w, h = [int(coord / detection_scale) for coord in face["box"]]
            face_image = frame[y : y + h, x : x + w]

            # Extract the embedding of the detected face
            try:
                embedding = DeepFace.represent(face_image, model_name=MODEL_NAME, model=model)[0]["embedding"]

                # Compare embedding with the stored ones
                distances = [np.linalg.norm(embedding - e) for e in embeddings]
                min_distance = min(distances)
                best_match = labels[np.argmin(distances)]

                label = best_match if min_distance < threshold else "Unknown"

                if show_boxes:
                    # Draw the box and label on the frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(
                        frame,
                        label,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
            except Exception as e:
                print(f"Error recognizing face: {e}")

        cv2.imshow("Real-Time Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting live feed.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and infer with DeepFace.")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode of operation: train or infer")

    train_parser = subparsers.add_parser("train", help="Train a model with labeled images.")
    train_parser.add_argument("--image-folder", type=str, default=DEFAULT_IMAGE_FOLDER, help="Path to the image folder.")
    train_parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Name of the model to save.")

    infer_parser = subparsers.add_parser("infer", help="Run real-time face recognition.")
    infer_parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Name of the model to load.")
    infer_parser.add_argument("--webcam", type=int, default=DEFAULT_WEBCAM_INDEX, help="Index of the webcam to use.")
    infer_parser.add_argument("--boxes", action="store_true", default=DEFAULT_SHOW_BOXES, help="Show bounding boxes.")
    infer_parser.add_argument("--scale", type=float, default=DEFAULT_DETECTION_SCALE, help="Scale factor for detection.")
    infer_parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Distance threshold for matching.")

    args = parser.parse_args()

    if args.mode == "train":
        train_classifier(args.image_folder, args.model_name)
    elif args.mode == "infer":
        recognize_faces(args.model_name, args.webcam, args.boxes, args.scale, args.threshold)