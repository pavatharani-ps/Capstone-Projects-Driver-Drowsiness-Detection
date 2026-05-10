import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights
from PIL import Image

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load Model
# =========================
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)

model.load_state_dict(torch.load("model.pth", map_location=device))

model = model.to(device)
model.eval()

# =========================
# Transform
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# Streamlit UI
# =========================
st.title("🚗 Driver Drowsiness Detection")

st.write("Upload a driver image to detect fatigue level.")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=300)

    # Preprocess image
    img = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():

        output = model(img)
        _, pred = torch.max(output, 1)

    # Display Result
    if pred.item() == 0:

        st.success("✅ ALERT - Driver is Awake")

    else:

        st.error("⚠️ FATIGUE DETECTED")