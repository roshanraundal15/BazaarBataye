import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# ------------------ CONFIG -------------------
# This path is correct because 'dataset' folder is a sibling to app.py
DATASET_PATH = "dataset"
QUALITY_LABELS = ["good", "average", "bad"]
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')

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

    # --- Debugging additions (these will show in Streamlit app logs) ---
    st.info(f"App's current working directory: {os.getcwd()}")
    st.info(f"Looking for dataset at: {os.path.abspath(DATASET_PATH)}")

    if not os.path.isdir(DATASET_PATH):
        st.error(f"Error: Base dataset directory '{os.path.abspath(DATASET_PATH)}' not found. "
                 "Ensure 'dataset' folder is committed to GitHub in 'image_detection/'.")
        return "Error", 0

    for quality in QUALITY_LABELS:
        quality_dir = os.path.join(DATASET_PATH, quality)
        st.info(f"Attempting to list directory: {quality_dir}") # Debug print

        if not os.path.isdir(quality_dir):
            st.error(f"Error: Quality reference directory '{quality_dir}' not found. "
                     "Make sure your 'dataset' folder has 'good', 'average', 'bad' subfolders.")
            return "Error", 0

        try:
            filenames = os.listdir(quality_dir)
            if not filenames:
                st.warning(f"Warning: Directory '{quality_dir}' is empty. No reference images found.")
                continue # Skip to next quality if directory is empty
        except Exception as e: # Catch any other exceptions during listing
            st.error(f"An unexpected error occurred when listing '{quality_dir}': {e}")
            return "Error", 0

        for filename in filenames:
            if filename.lower().endswith(IMG_EXTENSIONS):
                ref_img_path = os.path.join(quality_dir, filename)
                st.info(f"Loading reference image: {ref_img_path}") # Debug print
                ref_img = cv2.imread(ref_img_path)

                if ref_img is None: # Handle case where image fails to load
                    st.warning(f"Could not load reference image: {ref_img_path}. It might be corrupted or not a valid image.")
                    continue

                # Ensure images are the same size for comparison
                if uploaded_img.shape[0] > 0 and uploaded_img.shape[1] > 0:
                    ref_img = cv2.resize(ref_img, (uploaded_img.shape[1], uploaded_img.shape[0]))
                else:
                    st.warning("Uploaded image has invalid dimensions for resizing reference images.")
                    continue

                ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                ref_desc = extract_orb_features(ref_img)
                match_score = match_descriptors(uploaded_desc, ref_desc)

                if match_score > best_score:
                    best_score = match_score
                    predicted_quality = quality

    return predicted_quality, best_score

# ------------------ Fallback Explanation -------------------
def generate_explanation(quality_label):
    if quality_label == "good":
        return "This crop is of good quality. It is likely to fetch a higher price in the market and indicates healthy farming practices."
    elif quality_label == "average":
        return "This crop is of average quality. It may be sold at a standard price, but improvements in farming or storage could help."
    elif quality_label == "bad":
        return "This crop is of poor quality. It could reduce the market price and may indicate issues like pest damage, improper irrigation, or storage problems."
    else:
        return "Unable to provide explanation due to unknown quality."

# ------------------ Streamlit UI -------------------
st.set_page_config(page_title="Crop Quality Predictor", page_icon="🌾")
st.title("Quality Check")
st.write("Upload an image of a crop and get predicted quality using ORB feature matching.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict Quality"):
        label, score = predict_quality(image)
        if label == "Error":
            st.error("Prediction could not be completed. Please check the logs above for details.")
        else:
            st.success(f"🧠 Predicted Quality: **{label.upper()}** (Similarity Score: {score})")
            explanation = generate_explanation(label)
            st.info(f"📘 Explanation: {explanation}")
