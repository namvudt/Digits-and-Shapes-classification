"""
Gradio App for MNIST & Shapes Recognition
Combined app: MNIST + Shapes + Image Segmentation
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
import cv2
import numpy as np

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Shape classes
SHAPE_CLASSES = ['circle', 'hexagon', 'oval', 'rectangle', 'square', 'star', 'triangle']

# MNIST Model
class MNISTModel(nn.Module):
    def __init__(self, num_classes=10):
        super(MNISTModel, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# Shapes Model
class ShapesModel(nn.Module):
    def __init__(self, num_classes=7):
        super(ShapesModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# Load models
mnist_model = MNISTModel(num_classes=10).to(device)
shapes_model = ShapesModel(num_classes=7).to(device)

try:
    mnist_model.load_state_dict(torch.load('best_mnist_model.pth', map_location=device))
    mnist_model.eval()
    print("✓ Loaded MNIST model")
except:
    print("⚠️ Could not load MNIST model")

try:
    shapes_model.load_state_dict(torch.load('best_shapes_model_reduce.pth', map_location=device))
    shapes_model.eval()
    print("✓ Loaded Shapes model")
except:
    print("⚠️ Could not load Shapes model")


# Transforms
mnist_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

shapes_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Prediction functions
def process_canvas(canvas_data):
    """Process canvas data from Sketchpad (can be dict or numpy array)"""
    if canvas_data is None:
        return None
    
    # If it's a dict from Sketchpad, extract the composite image
    if isinstance(canvas_data, dict):
        if 'composite' in canvas_data and canvas_data['composite'] is not None:
            return canvas_data['composite']
        elif 'background' in canvas_data and canvas_data['background'] is not None:
            return canvas_data['background']
        else:
            return None
    
    # If it's already a numpy array, return as is
    return canvas_data


def predict_mnist(image):
    """Predict MNIST digit"""
    if image is None:
        return {"Error": 1.0}
    
    # Process canvas data if needed
    image = process_canvas(image)
    if image is None:
        return {"Error": 1.0}
    
    image = Image.fromarray(image).convert('L')
    image_tensor = mnist_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = mnist_model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
    
    probs = probabilities[0].cpu().numpy()
    result = {str(i): float(probs[i]) for i in range(10)}
    
    return result


def predict_shape(image):
    """Predict geometric shape"""
    if image is None:
        return {"Error": 1.0}
    
    # Process canvas data if needed
    image = process_canvas(image)
    if image is None:
        return {"Error": 1.0}
    
    image = Image.fromarray(image).convert('RGB')
    image_tensor = shapes_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = shapes_model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
    
    probs = probabilities[0].cpu().numpy()
    result = {SHAPE_CLASSES[i]: float(probs[i]) for i in range(7)}
    
    return result


def segment_image(image, method, k_value):
    """Segment image using traditional methods"""
    if image is None:
        return None, "Please upload an image"
    
    if method == "Otsu Threshold":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        info = "Applied Otsu's automatic threshold"
        
    elif method == "K-Means (Color)":
        pixel_values = image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixel_values, k_value, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        result = centers[labels.flatten()]
        result = result.reshape(image.shape)
        info = f"Applied K-Means segmentation with k={k_value}"
        
    elif method == "Edge Detection (Canny)":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        info = "Applied Canny edge detection"
        
    else:
        result = image
        info = "No segmentation applied"
    
    return result, info


# Create Gradio interface
with gr.Blocks(title="Vision AI - MNIST & Shapes", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 Vision AI - MNIST & Shapes
    ### MNIST Digits • Geometric Shapes • Draw & Predict • Image Segmentation
    """)
    
    with gr.Tabs():
        # Tab 1: Upload Image
        with gr.Tab("🔢 Upload & Classify"):
            gr.Markdown("### Upload image to classify digits or shapes")
            with gr.Row():
                with gr.Column():
                    img_classify = gr.Image(label="Upload Image", type="numpy")
                    task_type = gr.Radio(
                        choices=["MNIST Digit (0-9)", "Geometric Shape"],
                        value="MNIST Digit (0-9)",
                        label="Task Type"
                    )
                    btn_classify = gr.Button("Classify 🎯", variant="primary")
                
                with gr.Column():
                    output_classify = gr.Label(label="Predictions", num_top_classes=5)
            
            btn_classify.click(
                fn=lambda img, task: predict_mnist(img) if task == "MNIST Digit (0-9)" else predict_shape(img),
                inputs=[img_classify, task_type],
                outputs=output_classify
            )
        
        # Tab 2: Draw & Predict
        # with gr.Tab("✏️ Draw & Predict"):
        #     gr.Markdown("### Draw a digit (0-9) or shape and get instant prediction!")
        #     with gr.Row():
        #         with gr.Column():
        #             canvas = gr.Paint(
        #                 label="Draw Here - White pen on black background",
        #                 type="numpy",
        #                 image_mode="L",
        #                 canvas_size=(560, 560),
        #                 brush=gr.Brush(colors=["#FFFFFF"], default_size=15, color_mode="fixed")
        #             )
        #             draw_task_type = gr.Radio(
        #                 choices=["MNIST Digit (0-9)", "Geometric Shape"],
        #                 value="MNIST Digit (0-9)",
        #                 label="What are you drawing?"
        #             )
        #             with gr.Row():
        #                 btn_predict_draw = gr.Button("Predict 🎯", variant="primary")
        #                 btn_clear = gr.Button("Clear 🗑️", variant="secondary")
                
        #         with gr.Column():
        #             output_draw = gr.Label(label="Predictions", num_top_classes=5)
        #             gr.Markdown("""
        #             **Tips:**
        #             - Click the brush tool on the left
        #             - Draw large and centered
        #             - For digits: Draw like handwriting
        #             - For shapes: Draw clear outlines
        #             - Use Clear button or eraser to start over
        #             """)
            
        #     btn_predict_draw.click(
        #         fn=lambda img, task: predict_mnist(img) if task == "MNIST Digit (0-9)" else predict_shape(img),
        #         inputs=[canvas, draw_task_type],
        #         outputs=output_draw
        #     )
        #     btn_clear.click(fn=lambda: None, inputs=None, outputs=canvas)
        
        # Tab 3: Image Segmentation
        with gr.Tab("✂️ Image Segmentation"):
            gr.Markdown("### Segment images using traditional computer vision methods")
            with gr.Row():
                with gr.Column():
                    img_segment = gr.Image(label="Upload Image", type="numpy")
                    method_choice = gr.Radio(
                        choices=["Otsu Threshold", "K-Means (Color)", "Edge Detection (Canny)"],
                        value="K-Means (Color)",
                        label="Segmentation Method"
                    )
                    k_slider = gr.Slider(
                        minimum=2, maximum=10, value=3, step=1,
                        label="K value (for K-Means)"
                    )
                    btn_segment = gr.Button("Segment Image 🖼️", variant="primary")
                
                with gr.Column():
                    output_segment_img = gr.Image(label="Segmented Result")
                    output_segment_info = gr.Textbox(label="Segmentation Info", lines=3)
            
            btn_segment.click(
                fn=segment_image,
                inputs=[img_segment, method_choice, k_slider],
                outputs=[output_segment_img, output_segment_info]
            )
    
    gr.Markdown("""
    ---
    ### 📊 Model Information
    
    **Classification Models:**
    - MNIST: CNN trained on 60,000 handwritten digits (10 classes)
    - Shapes: CNN trained on geometric shapes (7 classes: circle, hexagon, oval, rectangle, square, star, triangle)
    
    **Image Processing:**
    - Segmentation: Traditional CV methods (Otsu, K-Means, Canny)
    
    **Device:** {}
    """.format(device))

# Launch
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Vision AI - MNIST & Shapes...")
    print("="*60)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
