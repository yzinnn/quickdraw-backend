import os, json
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import torch
from torch import nn
from torchvision import transforms

# ------------------ 모델 정의 ------------------ #
class DeepCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ------------------ Flask + 모델 로드 ------------------ #
app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scene_model_map = {}

for i in range(1, 4):
    scene_id = f"scene{i}"
    model_path = f"quickdraw_{scene_id}.pth"
    label_path = f"{scene_id}_class_labels.json"

    if not os.path.exists(model_path) or not os.path.exists(label_path):
        continue

    with open(label_path) as f:
        class_list = json.load(f)["classes"]

    model = DeepCNN(num_classes=len(class_list)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    scene_model_map[scene_id] = (model, class_list)

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x),
])

# ------------------ 예측 API ------------------ #
@app.route('/predict/<scene_id>', methods=['POST'])
def predict(scene_id):
    if scene_id not in scene_model_map:
        return jsonify({'error': 'Invalid scene ID'}), 400
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    model, class_list = scene_model_map[scene_id]
    img_bytes = request.files['image'].read()
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)
        pred = output.argmax().item()
        return jsonify({'label': class_list[pred]})

# ------------------ 서버 실행 ------------------ #
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
