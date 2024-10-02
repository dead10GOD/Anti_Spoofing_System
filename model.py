import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F
from skimage.feature import local_binary_pattern

# Define the same model as before
class ShallowCNN(nn.Module):
    def __init__(self):
        super(ShallowCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 112 * 112, 256)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 112 * 112)
        x = F.relu(self.fc1(x))
        return x

class DualChannelNet(nn.Module):
    def __init__(self):
        super(DualChannelNet, self).__init__()
        self.shallow_cnn = ShallowCNN()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Identity()  # Remove the last fully connected layer
        self.fc = nn.Linear(2048 + 256, 2)  # Combine deep and shallow features

    def forward(self, x):
        image, lbp_image = x
        deep_features = self.resnet50(image)
        shallow_features = self.shallow_cnn(lbp_image)
        combined_features = torch.cat((deep_features, shallow_features), dim=1)
        output = self.fc(combined_features)
        return output

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DualChannelNet()
model_path = 'C:\\Users\\KIIT\\Desktop\\project\\models\\dual_channel_model.pth'
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# Define the data transformations for prediction
data_transforms_predict = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def capture_image_from_camera():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Captured Image', frame)
        cv2.waitKey(1)  # Display the frame until any key is pressed
        image_path = 'captured_image.jpg'
        cv2.imwrite(image_path, frame)
    cap.release()
    cv2.destroyAllWindows()
    return image_path

def predict_image(image_path, model, device):
    image = Image.open(image_path).convert('RGB')
    image_transformed = data_transforms_predict(image).unsqueeze(0).to(device)
    lbp_image = extract_lbp(cv2.imread(image_path, 0)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model((image_transformed, lbp_image))
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return confidence.item(), 'real' if predicted.item() == 0 else 'spoof'

def extract_lbp(image):
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    lbp = lbp.astype(np.float32) / 255.0  # Normalize to [0, 1]
    lbp = cv2.resize(lbp, (224, 224))  # Resize to match input size of CNN
    lbp = np.expand_dims(lbp, axis=0)  # Add channel dimension
    return torch.from_numpy(lbp).float()

# Capture an image from the camera
image_path = capture_image_from_camera()

# Predict the captured image
confidence, prediction = predict_image(image_path, model, device)
print(f"Prediction: {prediction} with {confidence * 100:.2f}% confidence")
