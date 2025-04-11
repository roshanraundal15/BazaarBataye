import streamlit as st
import requests

# Page setup
st.set_page_config(page_title="AgriBot - Multilingual Chat", layout="centered")
st.title("🌾 AgriBot: कृषि सलाहकार चैटबॉट")
st.markdown("*बोलो हिंदी, मराठी या इंग्लिश में — AgriBot देगा सही सलाह!*")

# Function to build prompt
def build_prompt(user_message):
    return f"""
You are a helpful agricultural assistant for Indian farmers. 
You understand Hindi, Marathi, and English.
Answer in the same language as the question.
Avoid adding extra locations unless asked.
Keep the response clean, informative, and friendly — no emotional tags or unnecessary excitement.
no emojis
if i ask question in english give answer in english only
your name is AgriBot
User: {user_message}
Assistant:"""



# Function to call local LLaMA2 model via Ollama or similar
def call_llama(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama2",
                "prompt": prompt,
                "stream": False
            }
        )
        return response.json().get("response", "❌ जवाब नहीं मिला।")
    except Exception as e:
        return f"⚠ Error: {e}"

# Text input from user
user_input = st.text_area("पूछें अपना सवाल (Ask your agricultural question)", height=100)

# Button to generate answer
if st.button("📩 जवाब पाएं"):
    if user_input.strip():
        prompt = build_prompt(user_input)
        response = call_llama(prompt)
        st.markdown("### 🤖 AgriBot का जवाब:")
        st.success(response)
    else:
        st.warning("कृपया एक सवाल टाइप करें।")