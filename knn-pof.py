import numpy as np
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path


class CatDogKNNClassifier:
    def __init__(self, k=5, img_size=(64, 64)):
        """
        Initialize the KNN classifier for cat vs dog classification

        Args:
            k: Number of neighbors for KNN
            img_size: Tuple of (width, height) for image resizing
        """
        self.k = k
        self.img_size = img_size
        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.scaler = StandardScaler()
        self.is_trained = False

    def extract_features(self, image_path):
        """
        Extract features from an image using histogram and basic statistics

        Args:
            image_path: Path to the image file

        Returns:
            numpy array of features
        """
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Resize image
            img = cv2.resize(img, self.img_size)

            # Convert to different color spaces
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            features = []

            # Color histograms
            for i, color_img in enumerate([img, hsv]):
                for channel in range(3):
                    hist = cv2.calcHist([color_img], [channel], None, [32], [0, 256])
                    features.extend(hist.flatten())

            # Grayscale histogram
            gray_hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            features.extend(gray_hist.flatten())

            # Texture features using Local Binary Pattern approximation
            # Calculate gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            # Statistical features
            features.extend(
                [
                    np.mean(gray),
                    np.std(gray),
                    np.min(gray),
                    np.max(gray),
                    np.mean(grad_x),
                    np.std(grad_x),
                    np.mean(grad_y),
                    np.std(grad_y),
                    np.mean(img[:, :, 0]),
                    np.mean(img[:, :, 1]),
                    np.mean(img[:, :, 2]),  # RGB means
                    np.std(img[:, :, 0]),
                    np.std(img[:, :, 1]),
                    np.std(img[:, :, 2]),  # RGB stds
                ]
            )

            return np.array(features)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def load_dataset(self, dataset_path):
        """
        Load images from dataset directory structure:
        dataset_path/
        ├── cats/
        │   ├── cat1.jpg
        │   ├── cat2.jpg
        │   └── ...
        └── dogs/
            ├── dog1.jpg
            ├── dog2.jpg
            └── ...

        Args:
            dataset_path: Path to dataset directory

        Returns:
            X: Feature matrix
            y: Labels (0 for cats, 1 for dogs)
        """
        dataset_path = Path(dataset_path)
        X, y = [], []

        # Process cats (label 0)
        cats_path = dataset_path / "cats"
        if cats_path.exists():
            for img_file in cats_path.glob("*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    features = self.extract_features(img_file)
                    if features is not None:
                        X.append(features)
                        y.append(0)  # Cat label

        # Process dogs (label 1)
        dogs_path = dataset_path / "dogs"
        if dogs_path.exists():
            for img_file in dogs_path.glob("*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    features = self.extract_features(img_file)
                    if features is not None:
                        X.append(features)
                        y.append(1)  # Dog label

        return np.array(X), np.array(y)

    def train(self, dataset_path, test_size=0.2, random_state=42):
        """
        Train the KNN classifier

        Args:
            dataset_path: Path to training dataset
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        print("Loading dataset...")
        X, y = self.load_dataset(dataset_path)

        if len(X) == 0:
            raise ValueError("No images found in dataset. Check directory structure.")

        print(f"Loaded {len(X)} images")
        print(f"Cats: {np.sum(y == 0)}, Dogs: {np.sum(y == 1)}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train KNN
        print(f"Training KNN with k={self.k}...")
        self.knn.fit(X_train_scaled, y_train)

        # Evaluate
        train_pred = self.knn.predict(X_train_scaled)
        test_pred = self.knn.predict(X_test_scaled)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        print(f"\nTraining Accuracy: {train_acc:.3f}")
        print(f"Testing Accuracy: {test_acc:.3f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, test_pred, target_names=["Cat", "Dog"]))

        self.is_trained = True
        return train_acc, test_acc

    def predict_image(self, image_path):
        """
        Predict whether an image is a cat or dog

        Args:
            image_path: Path to image file

        Returns:
            prediction: 0 for cat, 1 for dog
            probability: Confidence of prediction
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        features = self.extract_features(image_path)
        if features is None:
            return None, None

        features_scaled = self.scaler.transform([features])
        prediction = self.knn.predict(features_scaled)[0]

        # Get probabilities (based on neighbor votes)
        probabilities = self.knn.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)

        return prediction, confidence

    def predict_and_show(self, image_path):
        """
        Predict and display the result with the image

        Args:
            image_path: Path to image file
        """
        prediction, confidence = self.predict_image(image_path)

        if prediction is None:
            print(f"Could not process image: {image_path}")
            return

        # Load and display image
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = "Dog" if prediction == 1 else "Cat"

        plt.figure(figsize=(8, 6))
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.title(f"Prediction: {label} (Confidence: {confidence:.3f})")
        plt.show()

        print(f"Predicted: {label} with confidence {confidence:.3f}")


def main():
    """
    Example usage of the CatDogKNNClassifier
    """
    # Initialize classifier
    classifier = CatDogKNNClassifier(k=5, img_size=(64, 64))

    # Example dataset path structure
    dataset_path = "knn-dataset"

    print("Cat vs Dog KNN Classifier")
    print("=" * 30)

    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset path '{dataset_path}' not found.")
        print("\nTo use this classifier:")
        print("1. Create a dataset directory with the following structure:")
        print("   dataset/")
        print("   ├── cats/")
        print("   │   ├── cat1.jpg")
        print("   │   └── ...")
        print("   └── dogs/")
        print("       ├── dog1.jpg")
        print("       └── ...")
        print("2. Update the 'dataset_path' variable with your dataset location")
        print("3. Run the script again")
        return

    try:
        # Train the model
        train_acc, test_acc = classifier.train(dataset_path)

        print(train_acc, test_acc)

        # Example prediction on a single image
        test_image = "image.jpg"
        if os.path.exists(test_image):
            classifier.predict_and_show(test_image)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
