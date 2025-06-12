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

# Define a threshold for what constitutes a "valid" match
# This value needs to be tuned based on your dataset and ORB performance.
# Start with a small value like 5-10 and adjust it.
# If you get too many "Unknowns", lower it. If you get wrong predictions, raise it.
SCORE_THRESHOLD = 5 # Minimum number of matches required to classify

# ------------------ ORB Feature Extractor -------------------
orb = cv2.ORB_create(nfeatures=500)

def extract_orb_features(image_rgb): # Expects an RGB image
    """
    Extracts ORB keypoints and descriptors from an RGB image.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return descriptors

def match_descriptors(desc1, desc2):
    """
    Matches ORB descriptors using Brute-Force Matcher.
    Returns the number of good matches.
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if desc1 is None or desc2 is None:
        return 0
    try:
        matches = bf.match(desc1, desc2)
        return len(matches)
    except cv2.error as e:
        # This can happen if descriptor types are incompatible or empty
        st.warning(f"Error during descriptor matching: {e}. Skipping match.")
        return 0


# ------------------ Prediction -------------------
def predict_quality(uploaded_img):
    # Convert uploaded PIL RGB image to OpenCV RGB format for consistency
    uploaded_cv_img_rgb = np.array(uploaded_img.convert('RGB')) # Ensure RGB

    # Extract features from uploaded image
    uploaded_desc = extract_orb_features(uploaded_cv_img_rgb)

    if uploaded_desc is None:
        st.warning("Warning: Could not extract features from the uploaded image. Please try another image.")
        return "Unknown", -1 # Cannot proceed without features

    best_score = 0 # Initialize best_score to 0, as match_descriptors returns non-negative
    predicted_quality = "Unknown"

    # --- Debugging additions (these will show in Streamlit app logs) ---
    st.info(f"App's current working directory: {os.getcwd()}")
    st.info(f"Looking for dataset at: {os.path.abspath(DATASET_PATH)}")

    if not os.path.isdir(DATASET_PATH):
        st.error(f"Error: Base dataset directory '{os.path.abspath(DATASET_PATH)}' not found. "
                 "Ensure 'dataset' folder is committed to GitHub in 'image_detection/'.")
        return "Error", -1 # Return -1 for score if directory missing

    for quality in QUALITY_LABELS:
        quality_dir = os.path.join(DATASET_PATH, quality)
        st.info(f"Attempting to list directory: {quality_dir}") # Debug print

        if not os.path.isdir(quality_dir):
            st.error(f"Error: Quality reference directory '{quality_dir}' not found. "
                     "Make sure your 'dataset' folder has 'good', 'average', 'bad' subfolders.")
            continue # Continue to next quality label, don't stop the whole prediction

        try:
            filenames = os.listdir(quality_dir)
            if not filenames:
                st.warning(f"Warning: Directory '{quality_dir}' is empty. No reference images found for this quality.")
                continue
        except Exception as e:
            st.error(f"An unexpected error occurred when listing '{quality_dir}': {e}")
            continue

        for filename in filenames:
            if filename.lower().endswith(IMG_EXTENSIONS):
                ref_img_path = os.path.join(quality_dir, filename)
                st.info(f"Loading reference image: {ref_img_path}") # Debug print
                ref_img_bgr = cv2.imread(ref_img_path) # Reads as BGR by default

                if ref_img_bgr is None:
                    st.warning(f"Could not load reference image: {ref_img_path}. Skipping.")
                    continue

                # Ensure image dimensions are valid before resizing
                if uploaded_cv_img_rgb.shape[0] == 0 or uploaded_cv_img_rgb.shape[1] == 0:
                    st.warning("Uploaded image has zero dimensions. Cannot resize reference images.")
                    continue

                # Resize reference image (BGR) to match uploaded image dimensions
                ref_img_bgr_resized = cv2.resize(ref_img_bgr, (uploaded_cv_img_rgb.shape[1], uploaded_cv_img_rgb.shape[0]))

                # Convert resized reference image to RGB before extracting features
                ref_img_rgb = cv2.cvtColor(ref_img_bgr_resized, cv2.COLOR_BGR2RGB)
                ref_desc = extract_orb_features(ref_img_rgb)

                if ref_desc is None:
                    st.warning(f"Warning: Could not extract features from reference image: {ref_img_path}. Skipping.")
                    continue

                match_score = match_descriptors(uploaded_desc, ref_desc)
                st.info(f"  Match score for {filename} ({quality}): {match_score}") # Detailed debug

                if match_score > best_score:
                    best_score = match_score
                    predicted_quality = quality

    # Final decision based on SCORE_THRESHOLD
    if best_score < SCORE_THRESHOLD:
        return "Unknown", best_score
    else:
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
