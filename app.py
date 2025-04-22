import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import snntorch as snn
from snntorch import surrogate

# Page Configuration
st.set_page_config(page_title="Wildlife SNN Classifier", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🦛 🐘 🦏 🦓 Wildlife Image Classifier using Spiking Neural Networks</h1>
    <p style='text-align: center;'>Buffalo | Elephant | Rhino | Zebra</p>
    <hr>
    """, unsafe_allow_html=True
)

# Layout: Split into two columns
left_col, right_col = st.columns(2)

# === LEFT COLUMN: Project Description ===
with left_col:
    st.subheader("📌 About This Project")
    st.markdown("""
    Welcome to our **Wildlife Image Classification App**! This project uses a **Spiking Neural Network (SNN)** built with PyTorch and snnTorch to classify images of four wild animals:
    
    - 🦛 **Buffalo**
    - 🐘 **Elephant**
    - 🦏 **Rhino**
    - 🦓 **Zebra**
    
    Unlike traditional neural networks, SNNs mimic how biological neurons work — processing data over **time steps** using spikes and membrane potential!

    ---
    
    🔬 **Tech Stack:**
    - PyTorch + snnTorch
    - Leaky Integrate-and-Fire (LIF) neurons
    - Streamlit for web interface

    🧪 Just upload an image and watch our model predict the animal type with high accuracy!
    """)

# === RIGHT COLUMN: Upload + Predict ===
with right_col:
    st.subheader("📷 Upload and Predict")

    CLASS_MAPPING = {
        0: 'Buffalo',
        1: 'Elephant',
        2: 'Rhino',
        3: 'Zebra'
    }

    # SNN Model compatible with your dataset structure
    class WildlifeSNN(nn.Module):
        def __init__(self, num_classes=4):
            super().__init__()
            beta = 0.95
            self.spike_grad = surrogate.fast_sigmoid()
            
            # Feature extraction
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.pool1 = nn.MaxPool2d(2)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=self.spike_grad)
            self.dropout1 = nn.Dropout(0.2)
            
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.pool2 = nn.MaxPool2d(2)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=self.spike_grad)
            self.dropout2 = nn.Dropout(0.2)
            
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.bn3 = nn.BatchNorm2d(64)
            self.pool3 = nn.AdaptiveAvgPool2d((4, 4))
            self.lif3 = snn.Leaky(beta=beta, spike_grad=self.spike_grad)
            self.dropout3 = nn.Dropout(0.2)
            
            # Classifier
            self.fc = nn.Linear(64*4*4, num_classes)
            self.lif_out = snn.Leaky(beta=beta, spike_grad=self.spike_grad, output=True)

        def forward(self, x, num_steps=50):
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem3 = self.lif3.init_leaky()
            mem_out = self.lif_out.init_leaky()
            spk_out_sum = 0
            
            for _ in range(num_steps):
                x1 = self.pool1(self.bn1(self.conv1(x)))
                spk1, mem1 = self.lif1(x1, mem1)
                
                x2 = self.pool2(self.bn2(self.conv2(spk1)))
                spk2, mem2 = self.lif2(x2, mem2)
                
                x3 = self.pool3(self.bn3(self.conv3(spk2)))
                spk3, mem3 = self.lif3(x3, mem3)
                
                x_flat = self.dropout3(spk3.view(spk3.size(0), -1))
                out = self.fc(x_flat)
                spk_out, mem_out = self.lif_out(out, mem_out)
                
                spk_out_sum += spk_out
            
            return spk_out_sum / num_steps


    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WildlifeSNN().to(device)
    model.load_state_dict(torch.load("snn_22.pth", map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=350)

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            pred_class = torch.argmax(output, dim=1).item()

        st.success(f"### 🧠 Predicted Class: **{CLASS_MAPPING[pred_class]}**")

        st.markdown("#### 🔢 Class Confidence Scores")
        for idx, score in enumerate(output.squeeze()):
            st.write(f"- {CLASS_MAPPING[idx]}: `{score.item():.4f}`")
