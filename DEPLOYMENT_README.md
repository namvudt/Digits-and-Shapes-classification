# MNIST & Shape Recognition Deployment

## Cài đặt

Cài Gradio (nếu chưa có):
```bash
pip install gradio
```

Hoặc cài toàn bộ dependencies:
```bash
pip install -r requirements_deploy.txt
```

## Chạy ứng dụng

```bash
python app_deploy.py
```

Sau khi chạy, mở trình duyệt và truy cập:
- **Local**: http://localhost:7860
- **Network**: http://0.0.0.0:7860

## Cách sử dụng

1. Upload ảnh (chữ số viết tay hoặc hình khối)
2. Chọn task type:
   - **MNIST Digit (0-9)**: Nhận dạng chữ số
   - **Geometric Shape**: Nhận dạng hình học (17 loại)
3. Click **Predict** để xem kết quả

## Models cần có

Đảm bảo có 2 file models trong thư mục:
- `best_mnist_model.pth` - Model nhận dạng chữ số
- `best_shapes_model.pth` - Model nhận dạng hình khối

## Hình khối được hỗ trợ

17 loại: circle, decagon, heptagon, hexagon, kite, nonagon, octagon, oval, parallelogram, pentagon, rectangle, rhombus, semicircle, square, star, trapezoid, triangle

## Tính năng

- ✅ Upload ảnh trực tiếp
- ✅ Xem top 5 predictions với confidence scores
- ✅ Hỗ trợ cả CPU và CUDA
- ✅ Interface thân thiện với Gradio
- ✅ Real-time prediction

## Công nghệ

- **Frontend**: Gradio
- **Backend**: PyTorch
- **Models**: Custom CNN architectures
