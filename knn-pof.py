import json
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os


class COCOKNNClassifier:
    def __init__(self, coco_json_path, images_dir, feature_model="resnet50"):
        """
        Args:
            coco_json_path: Path to COCO format JSON file
            images_dir: Directory containing the images
            feature_model: Pre-trained model to use for feature extraction
        """
        self.coco_json_path = coco_json_path
        self.images_dir = images_dir

        # Load COCO annotations
        with open(coco_json_path, "r") as f:
            self.coco_data = json.load(f)

        # Setup feature extraction model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        if feature_model == "resnet50":
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            # Remove the final classification layer
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        elif feature_model == "mobilenet":
            self.model = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.DEFAULT
            )
            self.model.classifier = torch.nn.Identity()

        self.model = self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Initialize KNN classifier
        self.knn = None
        self.label_encoder = LabelEncoder()

    def prepare_dataset(self, strategy="most_common", min_objects=1):
        """
        Prepare dataset from COCO annotations

        Args:
            strategy: 'most_common' (use most frequent class)
            min_objects: Minimum number of objects in image to include it

        Returns:
            image_paths: List of image file paths
            labels: List of class labels for each image
        """
        print("Preparing dataset from COCO annotations...")

        # Create category mapping
        categories_map = {
            cat["id"]: cat["name"] for cat in self.coco_data["categories"]
        }

        # Group annotations by image
        image_annotations = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann["category_id"])

        image_paths = []
        labels = []

        for img in self.coco_data["images"]:
            img_id = img["id"]

            if img_id not in image_annotations:
                continue

            # Get all category IDs for this image
            category_ids = image_annotations[img_id]

            if len(category_ids) < min_objects:
                continue

            if strategy == "most_common":
                most_common_category = max(set(category_ids), key=category_ids.count)
                label = categories_map[most_common_category]
            else:
                raise NotImplementedError(
                    "Only 'most_common' strategy is currently supported"
                )

            img_path = os.path.join(self.images_dir, img["file_name"])

            if os.path.exists(img_path):
                image_paths.append(img_path)
                labels.append(label)

        print(f"Found {len(image_paths)} images with annotations")
        print(f"Classes: {set(labels)}")

        return image_paths, labels

    def extract_features(self, image_paths):
        """
        Extract features from images using pre-trained CNN

        Args:
            image_paths: List of image file paths

        Returns:
            features: Numpy array of feature vectors
        """
        print("Extracting features from images...")
        features = []

        with torch.no_grad():
            for img_path in tqdm(image_paths):
                try:
                    # Load and preprocess image
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)

                    # Extract features
                    feature = self.model(img_tensor)
                    feature = feature.squeeze().cpu().numpy()
                    features.append(feature)

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    # Add zero vector for failed images
                    features.append(np.zeros(2048))  # ResNet50 feature size

        return np.array(features)

    def train(self, image_paths, labels, n_neighbors=5, test_size=0.2):
        """
        Train KNN classifier

        Args:
            image_paths: List of image paths
            labels: List of labels
            n_neighbors: Number of neighbors for KNN
            test_size: Fraction of data to use for testing
        """

        features = self.extract_features(image_paths)
        encoded_labels = self.label_encoder.fit_transform(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            encoded_labels,
            test_size=test_size,
            random_state=42,
            stratify=encoded_labels,
        )

        print(f"\nTraining set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")

        print(f"\nTraining KNN with {n_neighbors} neighbors...")
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        self.knn.fit(X_train, y_train)

        print("\nEvaluating on test set...")
        y_pred = self.knn.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=self.label_encoder.classes_,
                zero_division=0,
            )
        )

        return accuracy

    def predict(self, image_path):
        """
        Predict class for a single image

        Args:
            image_path: Path to image file

        Returns:
            predicted_class: Predicted class name
            confidence: Confidence score
        """
        if self.knn is None:
            raise Exception("Model not trained. Call train() first.")

        features = self.extract_features([image_path])

        prediction = self.knn.predict(features)[0]
        probabilities = self.knn.predict_proba(features)[0]

        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        confidence = probabilities.max()

        return predicted_class, confidence

    def save_model(self, save_path):
        """Save the trained KNN model and label encoder"""
        import pickle

        model_data = {"knn": self.knn, "label_encoder": self.label_encoder}
        with open(save_path, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {save_path}")

    def load_model(self, load_path):
        """Load a trained KNN model and label encoder"""
        import pickle

        with open(load_path, "rb") as f:
            model_data = pickle.load(f)
        self.knn = model_data["knn"]
        self.label_encoder = model_data["label_encoder"]
        print(f"Model loaded from {load_path}")


# Example usage
if __name__ == "__main__":

    classifier = COCOKNNClassifier(
        coco_json_path="coco-dataset/_annotations.coco.json",
        images_dir="coco-dataset",
        feature_model="resnet50",
    )

    image_paths, labels = classifier.prepare_dataset(strategy="most_common")
    accuracy = classifier.train(image_paths, labels, n_neighbors=5, test_size=0.2)

    # No need to save the model yet I guess??
    # classifier.save_model("knn_classifier.pkl")

    # Predict on a new image
    predicted_class, confidence = classifier.predict("image-for-knn.jpg")
    print(f"Predicted: {predicted_class} (confidence: {confidence:.4f})")
