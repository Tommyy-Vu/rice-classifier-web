<<<<<<< HEAD
# rice-classifier-web
Web phân loại gạo bằng học sâu (Flask + MobileNetV2)
=======
# Rice Authenticity Classification (MobileNetV2)

This project uses a fine-tuned **MobileNetV2** model to classify 10 rice types and determine whether a given sample is **Vietnamese**, **Foreign**, or **Unknown**.

## 🧠 Model
- Base: MobileNetV2 (from torchvision)
- Input size: 224x224
- Framework: PyTorch
- Trained on: Custom dataset of 10 rice types

## 🖥️ Web App
A simple **Flask** web interface to upload an image and view predictions.

### Run locally
```bash
pip install -r requirements.txt
python app.py
>>>>>>> 482e8e5 (Initial commit - rice classifier web)
