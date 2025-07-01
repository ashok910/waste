import torch
import torchvision.transforms as transforms
from PIL import Image
import os
class_names = full_dataset.classes
predict_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
def predict_waste_type(image_path, model, transform, class_names, device):
    model.eval()

    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None, None, None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None, None

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class_name = class_names[predicted_idx.item()]
    predicted_confidence = confidence.item()

    return predicted_class_name, predicted_confidence, predicted_idx.item()
if __name__ == '__main__':
    test_image_path = "/content/drive/MyDrive/Colab Notebooks/TrashType_Image_Dataset/trash/trash_003.jpg"

    if os.path.exists(test_image_path):
        predicted_type, confidence, predicted_index = predict_waste_type(
            test_image_path, model, predict_transform, class_names, device
        )

        if predicted_type:
            print(f"\n--- Prediction for: {test_image_path} ---")
            print(f"Predicted Waste Type: {predicted_type}")
            print(f"Confidence: {confidence:.4f}")
            print(f"Predicted Class Index: {predicted_index}")
        else:
            print("Prediction failed.")
    else:
        print(f"Test image not found at specified path: {test_image_path}")