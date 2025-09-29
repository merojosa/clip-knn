import torch
import clip
from PIL import Image

# Load the model
print(clip.available_models())
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define your two classes
class_names = ["cat", "dog"]  # <-- replace with your two classes

# Encode the text prompts
text_tokens = clip.tokenize(class_names).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Load and preprocess an image
image_path = "image2.jpg"
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# Encode the image
with torch.no_grad():
    image_features = model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)

# Compute similarity
similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
probs = similarities[0].cpu().numpy()

# Print results
for class_name, prob in zip(class_names, probs):
    print(f"{class_name}: {prob:.4f}")

print("Predicted class:", class_names[probs.argmax()])
