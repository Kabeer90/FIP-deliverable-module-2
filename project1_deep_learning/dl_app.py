"""
Automated Score Digitizer -- Streamlit Application
===================================================
Loads a pre-trained CNN (cnn_model.pth) and lets the user draw a digit on an
interactive canvas.  The model predicts the digit in real time.

Run with:
    streamlit run dl_app.py
"""

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# ---------------------------------------------------------------------------
# 1. MODEL ARCHITECTURE (must mirror the training notebook exactly)
# ---------------------------------------------------------------------------
class DigitCNN(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification.
    Architecture: 2 convolutional blocks followed by 3 fully-connected layers.
    """

    def __init__(self):
        super(DigitCNN, self).__init__()

        self.conv_block = nn.Sequential(
            # --- Block 1 ---
            # Input: 1x28x28  ->  Output: 32x14x14
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # --- Block 2 ---
            # Input: 32x14x14  ->  Output: 64x7x7
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_block = nn.Sequential(
            # Flatten 64*7*7 = 3136 features into a vector
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10),  # 10 output classes (digits 0-9)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x


# ---------------------------------------------------------------------------
# 2. MODEL LOADING (cached so it loads only once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(path: str = "cnn_model.pth") -> DigitCNN:
    """Load the trained CNN weights from disk."""
    model = DigitCNN()
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 3. IMAGE PREPROCESSING
# ---------------------------------------------------------------------------
def preprocess_canvas(canvas_result) -> torch.Tensor | None:
    """
    Convert the raw RGBA canvas image into a 1x1x28x28 tensor that matches
    the MNIST preprocessing pipeline used during training.

    Returns None if the canvas is blank.
    """
    if canvas_result is None or canvas_result.image_data is None:
        return None

    # The canvas returns an RGBA numpy array; convert to grayscale PIL image
    img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
    img = img.convert("L")  # convert to single-channel grayscale

    # Check if the canvas is essentially blank (all pixels near-white)
    img_array = np.array(img)
    if img_array.max() < 10:
        return None

    # Resize to 28x28 (MNIST native resolution)
    img = img.resize((28, 28), Image.LANCZOS)

    # Convert to float tensor, normalise to [0, 1], then apply MNIST statistics
    tensor = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
    tensor = (tensor - 0.1307) / 0.3081  # MNIST mean and std

    # Add batch and channel dimensions: (28,28) -> (1,1,28,28)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    return tensor


# ---------------------------------------------------------------------------
# 4. STREAMLIT UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Automated Score Digitizer", layout="centered")

    st.title("Automated Score Digitizer")
    st.markdown(
        "Draw a single digit (0-9) on the canvas below.  "
        "The CNN will predict the digit in real time."
    )

    # Load model
    model = load_model()

    # Two-column layout: canvas on the left, prediction on the right
    col_canvas, col_result = st.columns([2, 1])

    with col_canvas:
        st.subheader("Drawing Canvas")
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",  # transparent fill
            stroke_width=18,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="digit_canvas",
            update_streamlit=True,
            return_image_data=True,
        )

    with col_result:
        st.subheader("Prediction")

        tensor = preprocess_canvas(canvas_result)

        if tensor is not None:
            with torch.no_grad():
                logits = model(tensor)
                probabilities = torch.softmax(logits, dim=1).squeeze()
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class].item() * 100.0

            st.metric(label="Predicted Digit", value=str(predicted_class))
            st.metric(label="Confidence", value=f"{confidence:.1f}%")

            st.markdown("---")
            st.markdown("**Class Probabilities**")
            for digit in range(10):
                prob = probabilities[digit].item() * 100.0
                st.progress(prob / 100.0, text=f"{digit}: {prob:.1f}%")
        else:
            st.info("Draw a digit on the canvas to see a prediction.")


if __name__ == "__main__":
    main()
