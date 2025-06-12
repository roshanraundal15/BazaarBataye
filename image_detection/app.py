import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# ------------------ CONFIG -------------------
DATASET_PATH = "dataset"
QUALITY_LABELS = ["good", "average", "bad"]
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# Create dataset directory if it doesn't exist
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)
    for label in QUALITY_LABELS:
        os.makedirs(os.path.join(DATASET_PATH, label), exist_ok=True)
    st.warning(f"Dataset directory created at {DATASET_PATH}. Please add sample images for each quality category.")

# ------------------ ORB Feature Extractor -------------------
orb = cv2.ORB_create(nfeatures=500)

def extract_orb_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return descriptors

def match_descriptors(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if desc1 is None or desc2 is None:
        return 0
    matches = bf.match(desc1, desc2)
    return len(matches)

# ------------------ Prediction -------------------
def predict_quality(uploaded_img):
    uploaded_img = cv2.cvtColor(np.array(uploaded_img), cv2.COLOR_RGB2BGR)
    uploaded_desc = extract_orb_features(uploaded_img)

    best_score = -1
    predicted_quality = "Unknown"

    for quality in QUALITY_LABELS:
        quality_dir = os.path.join(DATASET_PATH, quality)
        if not os.path.exists(quality_dir):
            st.warning(f"Directory {quality_dir} not found. Skipping this quality category.")
            continue
            
        try:
            for filename in os.listdir(quality_dir):
                if filename.lower().endswith(IMG_EXTENSIONS):
                    ref_img_path = os.path.join(quality_dir, filename)
                    ref_img = cv2.imread(ref_img_path)
                    if ref_img is None:
                        st.warning(f"Could not read image: {ref_img_path}")
                        continue
                        
                    ref_img = cv2.resize(ref_img, (uploaded_img.shape[1], uploaded_img.shape[0]))
                    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                    ref_desc = extract_orb_features(ref_img)
                    match_score = match_descriptors(uploaded_desc, ref_desc)

                    if match_score > best_score:
                        best_score = match_score
                        predicted_quality = quality
        except Exception as e:
            st.error(f"Error processing {quality_dir}: {str(e)}")
            continue

    return predicted_quality, best_score if best_score != -1 else 0

# ------------------ Fallback Explanation -------------------
def generate_explanation(quality_label):
    explanations = {
        "good": "This crop is of good quality. It is likely to fetch a higher price in the market and indicates healthy farming practices.",
        "average": "This crop is of average quality. It may be sold at a standard price, but improvements in farming or storage could help.",
        "bad": "This crop is of poor quality. It could reduce the market price and may indicate issues like pest damage, improper irrigation, or storage problems.",
        "Unknown": "Unable to determine quality. The system may not have enough reference images for comparison."
    }
    return explanations.get(quality_label, explanations["Unknown"])

# ------------------ Streamlit UI -------------------
st.set_page_config(page_title="Crop Quality Predictor", page_icon="🌾")
st.title("Quality Check")
st.write("Upload an image of a crop and get predicted quality using ORB feature matching.")

# Check if dataset has images
has_images = False
for quality in QUALITY_LABELS:
    quality_dir = os.path.join(DATASET_PATH, quality)
    if os.path.exists(quality_dir) and any(f.lower().endswith(IMG_EXTENSIONS) for f in os.listdir(quality_dir)):
        has_images = True
        break

if not has_images:
    st.warning("⚠️ No reference images found in the dataset directory. Please add sample images to each quality folder for accurate predictions.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Quality"):
        try:
            label, score = predict_quality(image)
            st.success(f"🧠 Predicted Quality: **{label.upper()}** (Similarity Score: {score})")
            explanation = generate_explanation(label)
            st.info(f"📘 Explanation: {explanation}")
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.error("Please ensure you have reference images in the dataset folder.")
