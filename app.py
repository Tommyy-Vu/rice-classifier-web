import io, os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from flask import Flask, request, render_template, jsonify

# --------- Cấu hình ---------
MODEL_PATH = "mobilenetv2_model.pth"  # đặt cùng folder với app.py
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file model: {model_path}")

    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

# Cố gắng load model
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:", e)
    model = None

# --------- Transform ảnh ---------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --------- Hàm dự đoán ---------
def predict_image(img_bytes):
    if model is None:
        raise RuntimeError("Model chưa được load, vui lòng kiểm tra lại.")

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = int(probs.argmax())
    top_prob = float(probs[top_idx])
    label = classes[top_idx]

    # Xác định nhóm
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

# Trang chính
@app.route("/")
def index():
    return render_template("index.html")

# API dự đoán
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Không có file ảnh"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Tên file rỗng"}), 400

        img_bytes = file.read()
        result = predict_image(img_bytes)
        return jsonify(result), 200

    except Exception as e:
        print("⚠️ Error during prediction:", e)
        return jsonify({"error": str(e)}), 500

# --------- Bắt mọi lỗi Flask (để tránh trả về HTML) ---------
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Đường dẫn không tồn tại (404)"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Lỗi máy chủ (500)", "details": str(e)}), 500

# --------- Chạy app ---------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Flask server chạy tại http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
