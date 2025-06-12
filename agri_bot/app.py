import streamlit as st
import os
from dotenv import load_dotenv # Ensure this is at the top
import google.generativeai as genai # New import for direct Gemini API access

# Load environment variables from .env file (for local development)
# On deployment platforms like Cloud Run, these are set directly in the environment
load_dotenv()

# --- Google Gemini API Key Configuration ---
# Read API Key from environment variable
# IMPORTANT: Make sure GOOGLE_API_KEY is set in your .env file and deployment settings
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# --- Basic Validation for API Key ---
if not GOOGLE_API_KEY:
    st.error("Configuration Error: GOOGLE_API_KEY environment variable is not set. Please add it to your .env file or deployment settings.")
    st.stop() # Stop the app if API key is missing

# Configure the Gemini API client
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}. Check your GOOGLE_API_KEY.")
    st.stop()

# Load the Gemini model
try:
    # Use the specific model name, e.g., "gemini-pro" for text, or "gemini-1.5-flash-latest"
    # Note: "gemini-2.0-flash-001" might be specific to Vertex AI.
    # For direct API keys, common models are "gemini-pro" or "gemini-1.5-flash-latest"
    # Check the latest available models for direct API access on Google AI Studio documentation.
    model = genai.GenerativeModel("gemini-2.0-flash") # Using "gemini-pro" as a common default
except Exception as e:
    st.error(f"Error loading Gemini model: {e}. Check if the model is available via direct API key access.")
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

# Function to call Gemini model
def call_gemini(prompt):
    try:
        # Use the generate_content method for single-turn interactions
        response = model.generate_content(
            prompt,
            # Adjust temperature for creativity (higher = more creative, lower = more focused)
            generation_config={"temperature": 0.2, "max_output_tokens": 1024} # Max tokens for response length
        )
        # Access the text from the candidates
        return response.text
    except Exception as e:
        # Handle API errors, rate limits, etc.
        st.error(f"⚠ Error calling Gemini API: {e}")
        return "❌ जवाब नहीं मिला। Google Gemini API में कुछ समस्या है।"

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
