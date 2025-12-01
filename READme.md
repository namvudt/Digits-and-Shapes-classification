# 🔢🔷 Phân Loại Chữ Số và Hình Học

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Dự án Deep Learning phân loại chữ số viết tay (MNIST) và hình học 2D sử dụng kiến trúc CNN**

## 📋 Mục Lục

- [Giới Thiệu Dự Án](#giới-thiệu-dự-án)
- [Tính Năng](#tính-năng)
- [Công Nghệ Sử Dụng](#công-nghệ-sử-dụng)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [Cài Đặt](#cài-đặt)
- [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
- [Mô Hình](#mô-hình)
- [Kết Quả](#kết-quả)
- [Thành Viên Nhóm](#thành-viên-nhóm)
- [Ghi Nhận](#ghi-nhận)

## 🎯 Giới Thiệu Dự Án

Dự án này triển khai nhiều kiến trúc Convolutional Neural Network (CNN) để phân loại:
- **Chữ số viết tay** (0-9) từ bộ dữ liệu MNIST
- **Hình học 2D** (7 loại: circle, hexagon, oval, rectangle, square, star, triangle)


Dự án minh họa:
- ✅ Huấn luyện mô hình với cả **PyTorch** 
- ✅ Kỹ thuật tiền xử lý và tăng cường dữ liệu
- ✅ Nhiều kiến trúc mô hình 
- ✅ Triển khai web sử dụng **Gradio**
- ✅ Phương pháp xử lý ảnh truyền thống (phân đoạn)

## ✨ Tính Năng

### 🤖 Mô Hình
- **MNIST CNN Model** - Nhận dạng chữ số viết tay (10 lớp)
- **Shapes CNN Model** - Phân loại hình học (7 lớp)


### 🚀 Triển Khai
- **Gradio Web App** - Giao diện web tương tác
- **Giao diện đa tab**:
  - Upload & Phân loại
  - Vẽ & Dự đoán (canvas)
  - Phân đoạn ảnh (K-Means, Canny)


### 🔧 Xử Lý Ảnh
- Tăng cường dữ liệu (xoay, dịch chuyển, zoom)
- Hỗ trợ ảnh grayscale và màu
- Phương pháp CV truyền thống (phát hiện cạnh, phân cụm)

## 🛠️ Công Nghệ Sử Dụng

### Framework & Thư Viện
- **Deep Learning**: PyTorch
- **Computer Vision**: OpenCV, PIL/Pillow
- **Giao diện Web**: Gradio
- **Xử lý dữ liệu**: NumPy, Pandas, scikit-learn
- **Visualization**: Matplotlib, Seaborn


```

## 📁 Cấu Trúc Dự Án

Digits-and-Shapes-classification/src/
├── 📓 Notebooks
│   ├── mnist_features.ipynb           # MNIST cơ bản + features
│   ├── combined_recognition.ipynb     # Mô hình riêng (MNIST + Shapes)
│
├── 🚀 Triển Khai
│   ├── app_complete.py                # Ứng dụng Gradio chính (PyTorch)
│
├── 💾 Mô Hình
│   ├── best_mnist_model.pth           # Mô hình MNIST đã train
│   ├── best_shapes_model_reduce.pth   # Mô hình Shapes đã train
│
├── 📊 Dữ Liệu
│   ├── mnist/             # Dataset MNIST
│   │   ├── 0/
│   │   ├── 1/
│   │   ├── ...
│   │   └── 9/
│   └── 2D_Geometric_Shapes_Dataset/   # Dataset hình học
│       ├── circle/
│       ├── hexagon/
│       ├── oval/
│       ├── rectangle/
│       ├── square/
│       ├── star/
│       └── triangle/
│
└── 📄 Tài Liệu
    ├── README.md                      # File này
    ├── DEPLOYMENT_README.md           # Hướng dẫn triển khai PyTorch
    ├── requirements_deploy.txt        # Dependencies PyTorch
    |── BTL Xử Lý Ảnh.pdf              # Slide thuyết trình
```

## 🔧 Cài Đặt

### Yêu Cầu
- Python 3.8 trở lên
- CUDA 11.8+ (tùy chọn, cho GPU)

### Các Bước Cài Đặt

1. **Clone repository**
```bash
git clone https://github.com/namvudt/Digits-and-Shapes-classification.git
cd Digits-and-Shapes-classification
```

2. **Cài đặt thư viện**

Phiên bản PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install gradio pillow opencv-python numpy pandas matplotlib scikit-learn tqdm
```



Hoặc dùng file requirements:
```bash
pip install -r requirements_deploy.txt    # PyTorch
pip install -r requirements_keras.txt     # Keras
```

3. **Tải datasets**
- MNIST dataset :https://www.kaggle.com/datasets/shreyasi2002/corrupted-mnist
- 2D Geometric Shapes Dataset: https://www.kaggle.com/datasets/khalidboussaroual/2d-geometric-shapes-17-shapes

## 🚀 Hướng Dẫn Sử Dụng

### Huấn Luyện Mô Hình

**Train mô hình MNIST & Shapes (PyTorch):**
```bash
jupyter notebook combined_recognition.ipynb
jupyter mnist_feature.ipynb
# Chạy tất cả các cell để train cả 2 mô hình
```





### Chạy Ứng Dụng Web

**Phiên bản PyTorch:**
```bash
python app_complete.py
```
Truy cập tại: http://localhost:7860



### Dự Đoán

**Sử dụng giao diện Gradio:**
1. Mở ứng dụng web
2. Chọn một tab:
   - **Upload & Classify**: Upload ảnh chữ số hoặc hình học
   - **Draw & Predict**: Vẽ trực tiếp trên canvas
   - **Image Segmentation**: Áp dụng kỹ thuật CV
3. Chọn loại task:
   - MNIST Digit (0-9)
   - Geometric Shape
4. Nhận kết quả dự đoán với độ tin cậy



## 📊 Mô Hình

### So Sánh Mô Hình

| Mô Hình | Framework | Kích Thước Input | Số Lớp | Tham Số | Độ Chính Xác |
|---------|-----------|------------------|---------|---------|--------------|
| MNIST CNN | PyTorch | 28×28×1 | 10 | ~500K | ~99% |
| Shapes CNN | PyTorch | 64×64×3 | 7 | ~2M | ~95% |


### Chi Tiết Kiến Trúc

**MNIST Model:**
```python
Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool
→ Flatten → Dense(256) → Dropout(0.5) → Dense(10)
```

**Shapes Model:**
```python
Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool
→ Conv2D(256) → MaxPool → Flatten → Dense(512) → Dropout(0.5) → Dense(7)
```


## 📈 Kết Quả

### Hiệu Suất Huấn Luyện

**Mô Hình MNIST:**
- Độ chính xác Training: 99.5%
- Độ chính xác Validation: 99.2%
- Độ chính xác Test: 99.0%

**Mô Hình Shapes:**
- Độ chính xác Training: 98.0%
- Độ chính xác Validation: 95.5%
- Nhận dạng 7 hình học

### Ví Dụ Dự Đoán

```
Input: Chữ số '7' viết tay
Dự đoán: 7 (Độ tin cậy: 99.8%)

Input: Hình tròn
Dự đoán: circle (Độ tin cậy: 97.2%)

Input: Hình tam giác
Dự đoán: triangle (Độ tin cậy: 95.8%)
```

## 👥 Thành Viên Nhóm

Dự án được phát triển bởi nhóm 3 thành viên:

| Họ Tên | Đóng góp| 
|--------|---------|
| Nguyễn Nam Vũ| 40% | 
| Nguyễn Quyết Tiến| 30% |  
| Nguyễn Ngọc Thịnh | 30% |  






