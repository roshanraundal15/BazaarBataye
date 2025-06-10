import streamlit as st
import vertexai
from vertexai.preview.generative_models import GenerativeModel
import os

# --- Google Cloud Project Configuration ---
# Replace with your actual Google Cloud Project ID and Region
# Ensure you've authenticated correctly (gcloud auth application-default login locally, or service account for deployment)
PROJECT_ID = "baazar-bataye" # <--- THIS IS THE CRUCIAL CHANGE!
LOCATION = "us-central1" # Or your preferred region where Gemini is available (e.g., "asia-south1" if supported)

# Initialize Vertex AI
# It's good practice to do this once at the start of your app
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
except Exception as e:
    st.error(f"Error initializing Vertex AI: {e}. Make sure your Project ID and Location are correct and you're authenticated.")
    st.stop() # Stop the app if Vertex AI can't be initialized

# Load the Gemini model
try:
    model = GenerativeModel("gemini-2.0-flash-001") # Or "gemini-1.5-flash"
except Exception as e:
    st.error(f"Error loading Gemini model: {e}. Check if 'gemini-2.0' is available in {LOCATION} and your API is enabled.")
    st.stop()

# --- Streamlit App Setup ---
st.set_page_config(page_title="AgriBot - Multilingual Chat", layout="centered")
st.title("🌾 AgriBot: कृषि सलाहकार चैटबॉट")
st.markdown("*बोलो हिंदी, मराठी या इंग्लिश में — AgriBot देगा सही सलाह!*")

# Function to build prompt (can be more advanced with history later)
def build_prompt(user_message):
    return f"""
You are a helpful agricultural assistant for Indian farmers. 
You understand Hindi, Marathi, and English.
Answer in the same language as the question.
Avoid adding extra locations unless asked.
Keep the response clean, informative, and friendly — no emotional tags or unnecessary excitement.
no emojis
if i ask question in english give answer in english only
Your name is AgriBot.
User: {user_message}
Assistant:"""

# Function to call Gemini model via Vertex AI
def call_gemini(prompt):
    try:
        # Use the generate_content method for single-turn interactions
        # For multi-turn, you'd use `start_chat()`
        response = model.generate_content(
            prompt,
            # Adjust temperature for creativity (higher = more creative, lower = more focused)
            # You might want a lower temperature for factual agricultural advice
            generation_config={"temperature": 0.2, "max_output_tokens": 1024} # Max tokens for response length
        )
        # Access the text from the candidates
        return response.text
    except Exception as e:
        # Handle API errors, rate limits, etc.
        st.error(f"⚠ Error calling Gemini API: {e}")
        return "❌ जवाब नहीं मिला। Google Cloud API में कुछ समस्या है।"

# Text input from user
user_input = st.text_area("पूछें अपना सवाल (Ask your agricultural question)", height=100)

# Button to generate answer
if st.button("📩 जवाब पाएं"):
    if user_input.strip():
        with st.spinner("🤖 AgriBot सोच रहा है..."): # Show a spinner while processing
            prompt = build_prompt(user_input)
            response_text = call_gemini(prompt)
            st.markdown("### 🤖 AgriBot का जवाब:")
            st.success(response_text)
    else:
        st.warning("कृपया एक सवाल टाइप करें।")