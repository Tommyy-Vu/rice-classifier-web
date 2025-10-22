# app.py (Phiên bản đã sửa cho Hugging Face)
import io, os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from flask import Flask, request, render_template, jsonify

# --------- Cấu hình ---------
MODEL_PATH = "mobilenetv2_model.pth"  # đặt cùng folder với app.py
DEVICE = torch.device("cpu") # <--- SỬA 1: Ép dùng CPU (vì HF Space Free không có GPU)
THRESHOLD_UNKNOWN = 0.60  # nếu prob < threshold => "Không xác định"

# --------- Tên lớp ---------
classes = [
    "Ipsala", "gaosengcu", "Arborio", "gaoxideoBacha",
    "Karacadag", "nepcaihoavang", "Jasmine", "tamthai",
    "Basmati", "gaoST25"
]

# --------- Nhóm gạo ---------
vn_classes = {"gaosengcu", "gaoxideoBacha", "nepcaihoavang", "tamthai", "gaoST25"}
foreign_classes = {"Ipsala", "Arborio", "Karacadag", "Jasmine", "Basmati"}

# --------- Load model ---------
def load_model(model_path, num_classes=len(classes)):
    # <--- SỬA 2: Dùng tiêu chuẩn 'weights=None' thay vì 'pretrained=False'
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    # <--- SỬA 3: Thêm kiểm tra file rõ ràng để debug
    if not os.path.exists(model_path):
        error_msg = f"❌ LỖI NGHIÊM TRỌNG: Không tìm thấy file model tại: {model_path}"
        print(error_msg)
        raise FileNotFoundError(error_msg)

    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

print("Đang tải model...")
model = load_model(MODEL_PATH)
print("✅ Model đã tải thành công!")

# --------- Xử lý ảnh ---------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --------- Dự đoán ---------
def predict_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = int(probs.argmax())
    top_prob = float(probs[top_idx])
    label = classes[top_idx]

    # Phân loại nhóm gạo
    if top_prob < THRESHOLD_UNKNOWN:
        group = "Không xác định"
        color = "gray"
    elif label in vn_classes:
        group = "Gạo Việt Nam"
        color = "green"
    elif label in foreign_classes:
        group = "Gạo nước ngoài"
        color = "blue"
    else:
        group = "Không xác định"
        color = "gray"

    return {
        "label": label,
        "prob": round(top_prob, 4),
        "group": group,
        "color": color,
        "probs": {classes[i]: float(probs[i]) for i in range(len(classes))}
    }

# --------- Flask app ---------
app = Flask(__name__, static_folder="static")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Không có file ảnh"}), 400
    file = request.files["file"]
    img_bytes = file.read()
    try:
        res = predict_image(img_bytes)
        return jsonify(res)
    except Exception as e:
        print(f"⚠️ Lỗi khi dự đoán: {e}") # In lỗi ra log của HF
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Dòng này chỉ chạy khi bạn chạy local (python app.py)
    # Khi deploy, Gunicorn sẽ chạy file này, không qua __main__
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
