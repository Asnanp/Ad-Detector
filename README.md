🚀 Ultimate AI Ad Detection & Analysis System
Detect advertisements in text and images with cutting-edge machine learning and OCR — all in one powerful web app!

🎯 Project Overview
This web application leverages advanced machine learning and natural language processing to analyze both text and images for detecting advertisements. It combines:

Text classification models with comprehensive linguistic and sentiment features

Optical Character Recognition (OCR) on images to extract embedded text

Multi-layered analysis including danger detection, emotion analysis, scam detection, and naturalness assessment

This system helps identify hidden ads, phishing attempts, or scammy content for better content moderation and user safety.

🌟 Features
📝 Text Analysis: Input any text and get instant advertisement detection results with confidence scoring

🖼️ Image Analysis: Upload images; the app extracts text via OCR and analyzes for ads

🚦 Danger Level: Evaluates risk and flags suspicious or harmful content

🎭 Emotion Analysis: Detects emotional tone and intensity in the text

🔍 Scam Detection: Flags common scam types with confidence metrics

😊 Naturalness Assessment: Determines if the content feels artificially generated or human-like

⚡ Real-time Feedback: Fast results with loading animations and dynamic UI

📱 Responsive UI: Works smoothly on desktop and mobile

🔒 Secure File Handling: Image uploads validated for safety

RESTful API: Supports programmatic access for integration in other systems

🛠️ Tech Stack
Layer	Technology / Library
Backend	Python, Flask
ML Models	Scikit-learn (Logistic Regression)
NLP	NLTK (Sentiment, VADER), TextBlob
OCR	EasyOCR
Frontend	HTML5, CSS3, Vanilla JavaScript
Deployment	Docker, Heroku / AWS / VPS

🏗️ Architecture
Frontend
Responsive, modern UI using semantic HTML and CSS, powered by JavaScript for asynchronous calls and real-time results display.

Backend
Flask REST API endpoints for text and image analysis. Preprocesses input, runs ML predictions, and returns JSON results.

Machine Learning
Pre-trained models loaded on startup for instant prediction:

Text classification with engineered features (sentiment, POS tags, readability)

OCR text extraction from images using EasyOCR

Security & Validation
Sanitizes inputs, restricts upload file types, and handles errors gracefully.

📦 Installation & Setup
Requirements
Python 3.8+

4GB+ RAM for OCR

pip package manager

Step-by-step
bash
Copy
Edit
# Clone repo
git clone https://github.com/YourUsername/ad-prediction-webapp.git
cd ad-prediction-webapp

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data for sentiment
python -c "import nltk; nltk.download('vader_lexicon')"

# Verify model files are present:
# - text_model.pkl
# - text_vectorizor.pkl
# - text_scaler.pkl

# Run app locally
python app.py
Open your browser at http://localhost:5000

💻 Usage Guide
Web Interface
Text Analysis

Navigate to the "📝 Text Analysis" tab

Paste or type your text

Click Analyze Text

View detailed results with confidence, danger score, emotion, scam type, and naturalness

Image Analysis

Go to "🖼️ Image Analysis" tab

Upload image (supports JPG, PNG, GIF, WebP)

Click Analyze Image

View extracted text with bounding boxes and ad detection results

API Endpoints
Endpoint	Method	Description	Payload Example
/predict_text	POST	Analyzes text for ads	{ "text": "your input text" }
/predict_image	POST	Analyzes image for ads (multipart)	image: <file>
/health	GET	Server health check	N/A

🔬 Model Details
Type: Logistic Regression (best accuracy)

Features: 22+ linguistic and sentiment-based features, including TF-IDF vectorized text

Training Dataset: Curated ads and non-ads text samples

Accuracy: ~99.4% on test set

OCR: EasyOCR extracts text from images for text classification

⚙️ Deployment Options
Local
Run via Flask development server (for testing only)

Docker
dockerfile
Copy
Edit
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
Build & run:

bash
Copy
Edit
docker build -t ad-prediction-app .
docker run -p 5000:5000 ad-prediction-app
Cloud
Heroku: Use Procfile with web: python app.py

AWS/GCP: Deploy using Docker containers or serverless functions

VPS: Setup Gunicorn + Nginx reverse proxy for production-grade service

🛠️ Troubleshooting
Issue	Solution
ModuleNotFoundError	Run pip install -r requirements.txt
Missing model files	Confirm .pkl files exist in project root
OCR slow or fails	Check RAM, use high-quality images, verify internet for EasyOCR first run
Port 5000 busy	Kill conflicting process or change port in app.py
API returns error	Check JSON payload, enable debug logs

🔐 Security Considerations
Strict file type checking on uploads

Sanitize user input to prevent injection attacks

For production, enforce HTTPS, API auth, and rate limiting

Log errors securely for audit trail

📸 Screenshots & Demo
(Add screenshots or GIFs here for better UX understanding)

📈 Performance & Limitations
OCR accuracy depends on image quality and clarity

Models trained on specific ad datasets; may require retraining for new domains

Real-time analysis optimized for small to medium text and image sizes

📚 References & Resources
Scikit-learn Documentation

EasyOCR GitHub

NLTK VADER Sentiment Analysis

🤝 Contributing
Contributions welcome! Feel free to:

Open issues

Submit pull requests

Suggest new features or improvements

📄 License
MIT License - free to use for personal, educational, and commercial projects with attribution.

📞 Support
Facing issues? Reach out via GitHub Issues or email at your-email@example.com
