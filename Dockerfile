# Sử dụng ảnh Python 3.10
FROM python:3.10-slim

# Đặt thư mục làm việc bên trong container
WORKDIR /app

# Cài đặt các thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code (app.py, templates/, model.pth) vào
COPY . .

# HF Spaces chạy app trên cổng 7860
ENV PORT 7860
EXPOSE 7860

# Lệnh để chạy app bằng Gunicorn (tăng timeout lên 300s)
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "300", "app:app"]
