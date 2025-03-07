import streamlit as st
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load your model (replace with your actual model path)
model_path = "/Users/Cs-Store/Desktop/intern2/OCRME/fine_tuned_trocr_khmer"  
processor = TrOCRProcessor.from_pretrained(model_path)
model = VisionEncoderDecoderModel.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def perform_ocr(image):
    """Performs OCR on the given image."""
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return predicted_text

# Streamlit UI
st.title("Khmer OCR with Feedback")

uploaded_file = st.file_uploader("Upload an image", type=["png"])  # PNG only for now

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    predicted_text = perform_ocr(image)

    # Display prediction with Khmer font
    st.markdown(f'<p style="font-family: Khmer OS Battambang;">**Prediction:** {predicted_text}</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Correct"):
            st.success("Thank you for the feedback!")
    with col2:
        if st.button("Incorrect"):
            st.error("Please provide feedback to improve the model.")
