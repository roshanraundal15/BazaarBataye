Bazaar Bataye
Project Overview
Bazaar Bataye is a comprehensive agricultural intelligence platform designed to empower Indian farmers and agricultural stakeholders. It offers a user-friendly frontend, integrated with powerful backend services that leverage data analytics, machine learning for forecasting, computer vision for crop quality assessment, and cutting-edge AI for conversational assistance. The platform aims to provide valuable insights and recommendations to optimize farming practices and market decisions.

The project is structured with a frontend application (built with modern web technologies) and a backend/analytical Streamlit application (which handles the core AI and data processing functionalities).
![WhatsApp Image 2025-06-05 at 14 13 50_d7872147](https://github.com/user-attachments/assets/b45ff909-21a6-4337-9858-aa3a2c36c8b3)

Features
Bazaar Bataye provides the following core functionalities:

1. 📈 Agri Commodity Forecasting
Market Price Prediction: Forecasts future prices of various agricultural commodities in different districts of Maharashtra using the Prophet time series model.
Profit Projection: Calculates potential profit based on user-defined investment and yield.
Historical Trends: Visualizes historical monthly average prices.
AI-ML Based Recommendations: Offers actionable advice (Hold, Sell) based on forecasted prices and thresholds, accompanied by AI-generated reasoning considering soil quality, location trends, and supply-demand factors.
Interactive Visualizations: Presents price trends and forecasts using interactive Plotly charts.
![image](https://github.com/user-attachments/assets/782f72ca-f2de-4608-a9d0-4b492a811e79)
![image](https://github.com/user-attachments/assets/15e293bf-ef5a-42f3-9722-b5902719a5c3)


2. 🌾 Crop Quality Predictor
Image-Based Quality Assessment: Upload an image of a crop, and the system predicts its quality (Good, Average, Bad) using ORB (Oriented FAST and Rotated BRIEF) feature matching against a pre-defined dataset.
Similarity Score: Displays a similarity score indicating the confidence of the prediction.
Quality Explanation: Provides a textual explanation of the predicted quality and its implications.
![image](https://github.com/user-attachments/assets/09bfb6fe-7249-41aa-8d55-74679ffde606)


3. 🤖 AgriBot - Multilingual Chat
AI-Powered Agricultural Advisor: A conversational chatbot powered by Google's Gemini model (via Vertex AI), designed to answer farmers' questions.
Multilingual Support: Understands and responds in Hindi, Marathi, and English.
![image](https://github.com/user-attachments/assets/b8d94e46-754c-461d-8ea7-f6de36cd7b00)
![image](https://github.com/user-attachments/assets/3e8422dd-c3c1-4d9b-8a8e-52ff5591d210)


Context-Aware Advice: Provides clean, informative, and friendly advice relevant to Indian agriculture.
Technologies Used
This project utilizes a blend of modern frontend and robust backend technologies:

Frontend
Framework: React
Build Tool: Vite
Language: TypeScript
UI Components: shadcn-ui
Styling: Tailwind CSS
Backend/Analytics (Streamlit Application)
Web Framework: Streamlit
Data Manipulation: Pandas, NumPy
Time Series Forecasting: Prophet (from Meta)
Data Visualization: Plotly (Graph Objects and Express)
AI/ML Models: Google Gemini (via Google Cloud Vertex AI)
Image Processing: OpenCV (cv2), PIL (Pillow)
HTTP Requests: requests library (potentially for frontend-backend communication or external APIs)
PDF Generation (Future): ReportLab (imported, suggesting future report generation features)
Setup and Installation
This project consists of two main parts: a frontend application and a Streamlit-based backend/analytics application.

Prerequisites
Python 3.8+
Node.js & npm (for the frontend)
A Google Cloud Project with the Vertex AI API enabled.
A Service Account with the 'Vertex AI User' role for your Google Cloud Project.
A dataset for crop quality prediction in the dataset directory (structured as dataset/good/, dataset/average/, dataset/bad/ with reference images inside).
A CSV data file named maharashtra_market_daily_complete.csv containing commodity price data.
1. Frontend Setup
A. Local Development:

Clone the repository:
Bash

git clone <YOUR_GIT_URL> # Replace with your project's Git URL
cd <YOUR_PROJECT_NAME> # Replace with your project's directory name
Install dependencies:
Bash

npm install
Start the development server:
Bash

npm run dev
This will launch the frontend with auto-reloading and an instant preview.
B. Edit directly in GitHub:

Navigate to the desired file(s) in your GitHub repository.
Click the "Edit" button (pencil icon) at the top right of the file view.
Make your changes and commit them.
C. Use GitHub Codespaces:

From your repository's main page, click the "Code" button (green).
Select the "Codespaces" tab and click "New codespace" to launch a new environment.
Edit files directly within Codespaces, then commit and push your changes.
2. Streamlit Application Setup (Backend/Analytics)
The Streamlit application will likely run separately and potentially be integrated with the frontend via API calls or embedded if your architecture allows.

Navigate to the Streamlit app directory:
Assuming your Streamlit app code is in the root or a specific subdirectory, change into it. For example:

Bash

cd Bazaar-Bataye/agri_commodity_forecasting # Or the directory where your main Streamlit app.py is
(Note: If all your Streamlit code is in a single app.py in the root, you might not need to cd into a subdirectory).


Install Python dependencies:
Create a requirements.txt file in the root of your Streamlit app's directory (or the main project root if it's a single app.py) with the following content:

streamlit
pandas
numpy
plotly
prophet
google-cloud-aiplatform # For Vertex AI and Gemini access
opencv-python # For image processing (cv2)
Pillow # For image handling (PIL)
requests
reportlab
Then install:

Bash

pip install -r requirements.txt
Place data and model files:

Ensure maharashtra_market_daily_complete.csv is in the same directory as your Streamlit app.py (or correctly relative to it).
Place your crop quality prediction dataset directory (containing good, average, bad subfolders with images) in the same location as the relevant Streamlit app.py for quality prediction.


Bash

streamlit run app.py
(If you have multiple Streamlit app.py files for different functionalities, run them separately, e.g., streamlit run agri_commodity_forecasting/app.py, or structure your main app.py to navigate between features.)

Deployment
1. Deploying the Frontend
Instructions for deploying your frontend will depend on your chosen hosting provider (e.g., Vercel, Netlify, GitHub Pages, etc.).

2. Deploying the Streamlit Application (Backend/Analytics)
The Streamlit Community Cloud is the highly recommended platform for deploying the Streamlit part of this application.

Ensure GitHub Repository is Ready:

Make sure all your Streamlit application files, including app.py, maharashtra_market_daily_complete.csv, the dataset directory with its images, and requirements.txt, are committed and pushed to your GitHub repository.
CRITICAL: Do NOT commit your Google Cloud Service Account JSON key file or hardcoded API keys to GitHub.
Configure Google Cloud Service Account as a Streamlit Secret:

For the value, copy and paste the entire content of your Google Cloud Service Account JSON key file (downloaded during prerequisites) into the value field. This will securely provide your app with the necessary credentials to interact with Vertex AI/Gemini.
Set Google Cloud Project ID and Location in app.py:

Ensure your app.py correctly defines PROJECT_ID = "baazar-bataye" and LOCATION = "us-central1" (or your chosen region) to match your Google Cloud Project.
Deploy/Reboot App:

Once your GitHub repository is updated with the requirements.txt and all necessary files, and your Google Cloud secret is configured in Streamlit Cloud, the platform will automatically redeploy your app. You can also manually trigger a "Reboot app" from the Streamlit Cloud dashboard.
Project Structure (Recommended for Combined Functionality)
For better organization and maintainability, especially with multiple distinct functionalities within a single Streamlit app, consider structuring your project like this, or by splitting into multiple dedicated Streamlit apps if preferred:

Bazaar-Bataye/
├── .streamlit/
│   ├── config.toml      # Optional: For Streamlit UI configurations
│   └── secrets.toml     # (Local only, DO NOT COMMIT) For local secrets like GOOGLE_API_KEY
├── agri_commodity_forecasting/
│   └── app.py           # Main app for forecasting (or a module if integrated)
│   └── maharashtra_market_daily_complete.csv
├── crop_quality_predictor/
│   └── app.py           # Main app for quality prediction (or a module if integrated)
│   └── dataset/         # Contains reference images for quality prediction
│       ├── good/
│       ├── average/
│       └── bad/
├── agribot_chat/
│   └── app.py           # Main app for the chatbot (or a module if integrated)
├── frontend/            # Your React frontend project files
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
├── requirements.txt     # All Python dependencies for Streamlit apps
├── README.md
└── .gitignore           # Important: Include 'secrets.toml', 'service-account-key.json', 'venv*', 'node_modules'
If your current app.py directly contains all three functionalities, you might consider refactoring it into separate Python files within logical subdirectories and then importing them into a main app.py that acts as a central orchestrator or uses Streamlit's multi-page app feature.
