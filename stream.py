import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load Model
# =========================
model = models.mobilenet_v2(weights=None)

# 4 classes
model.classifier[1] = nn.Linear(model.last_channel, 4)

# Load saved model
model.load_state_dict(
    torch.load("model.pth", map_location=device)
)

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
# Class Names
# =========================
classes = ['Closed', 'Open', 'no-yawn', 'yawn']

# =========================
# Streamlit UI
# =========================
st.title("🚗 Driver Drowsiness Detection")

st.write(
    "Upload a driver image to detect fatigue level."
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# Prediction
# =========================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=300
    )

    # Preprocess image
    img = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():

        outputs = model(img)

        _, pred = torch.max(outputs, 1)

    predicted_class = classes[pred.item()]

    st.write(f"### Predicted Class: {predicted_class}")

    # =========================
    # Fatigue Decision Logic
    # =========================

    if predicted_class in ['Open', 'no-yawn']:

        st.success("✅ ALERT")

    elif predicted_class == 'yawn':

        st.warning("😴 MILD FATIGUE")

    elif predicted_class == 'Closed':

        st.error("⚠️ SEVERE FATIGUE")