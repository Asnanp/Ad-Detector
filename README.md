# 🎯 Ad Prediction Web Application

A comprehensive web application that uses machine learning to detect advertisements in both text and images. The system combines text analysis with OCR (Optical Character Recognition) to provide accurate ad detection capabilities.

## 🌟 Features

- **Text Analysis**: Analyze any text input to determine if it's an advertisement
- **Image Analysis**: Upload images and extract text using OCR, then analyze for ad content
- **Real-time Predictions**: Get instant results with confidence scores
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Advanced ML Models**: Uses trained scikit-learn models with comprehensive feature extraction

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- At least 4GB RAM (for OCR processing)

### Installation

1. **Clone or download the project files**
   ```bash
   # Make sure you have these files in your directory:
   # - app.py
   # - templates/index.html
   # - requirements.txt
   # - text_model.pkl
   # - text_vectorizor.pkl
   # - text_scaler.pkl
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv ad_prediction_env
   
   # On Windows:
   ad_prediction_env\Scripts\activate
   
   # On macOS/Linux:
   source ad_prediction_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data (required for sentiment analysis)**
   ```python
   python -c "import nltk; nltk.download('vader_lexicon')"
   ```

5. **Verify model files exist**
   Make sure these files are in your project directory:
   - `text_model.pkl` - The trained text classification model
   - `text_vectorizor.pkl` - The TF-IDF vectorizer
   - `text_scaler.pkl` - The feature scaler

### Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your web browser**
   Navigate to: `http://localhost:5000`

3. **Start using the application!**
   - Use the "Text Analysis" tab to analyze text
   - Use the "Image Analysis" tab to upload and analyze images

## 📱 How to Use

### Text Analysis
1. Click on the "📝 Text Analysis" tab
2. Enter or paste your text in the text area
3. Click "🔍 Analyze Text"
4. View the results with confidence score

### Image Analysis
1. Click on the "🖼️ Image Analysis" tab
2. Upload an image by clicking the upload area or dragging and dropping
3. Click "🔍 Analyze Image"
4. View the extracted text and prediction results

## 🔧 API Endpoints

The application provides REST API endpoints for programmatic access:

### Text Prediction
```http
POST /predict_text
Content-Type: application/json

{
    "text": "Your text here"
}
```

### Image Prediction
```http
POST /predict_image
Content-Type: multipart/form-data

image: [image file]
```

### Health Check
```http
GET /health
```

## 🏗️ Architecture

### Backend Components
- **Flask Web Server**: Handles HTTP requests and serves the web interface
- **ML Models**: Pre-trained scikit-learn models for text classification
- **OCR Engine**: EasyOCR for text extraction from images
- **Feature Extraction**: Comprehensive text analysis including sentiment, keywords, and linguistic features

### Frontend Components
- **Responsive HTML/CSS**: Modern, mobile-friendly interface
- **JavaScript**: Handles form submissions, file uploads, and result display
- **Real-time Updates**: Dynamic loading indicators and result animations

## 🛠️ Troubleshooting

### Common Issues

1. **"Module not found" errors**
   - Make sure you've installed all requirements: `pip install -r requirements.txt`
   - Verify you're using the correct Python environment

2. **Model files not found**
   - Ensure `text_model.pkl`, `text_vectorizor.pkl`, and `text_scaler.pkl` are in the project directory
   - Check file permissions

3. **OCR not working**
   - EasyOCR requires internet connection for first-time setup
   - Ensure sufficient RAM (4GB+) for OCR processing

4. **Port already in use**
   - Change the port in `app.py`: `app.run(port=5001)`
   - Or kill the process using port 5000

### Performance Tips

- **For better OCR performance**: Use high-quality, high-contrast images
- **For faster loading**: Keep image files under 5MB
- **For production**: Set `debug=False` in `app.py`

## 🔒 Security Considerations

- File upload validation is implemented for image types
- Input sanitization for text analysis
- For production deployment, consider:
  - Adding authentication
  - Implementing rate limiting
  - Using HTTPS
  - Setting up proper error logging

## 📊 Model Information

The application uses enhanced machine learning models trained on advertisement data:

- **Text Features**: 22+ comprehensive features including sentiment, keywords, linguistic patterns
- **Model Type**: Logistic Regression (best performing from multiple tested models)
- **Accuracy**: ~99.4% on test data
- **Preprocessing**: Advanced text cleaning and feature extraction

## 🚀 Deployment Options

### Local Development
- Use the built-in Flask development server (as shown above)

### Production Deployment
- **Heroku**: Add `Procfile` with `web: python app.py`
- **AWS/GCP**: Use container deployment with Docker
- **VPS**: Use Gunicorn with Nginx reverse proxy

### Docker Deployment
Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## 📝 License

This project is for educational and research purposes. Please ensure you have the right to use the trained models in your specific use case.

## 🤝 Contributing

Feel free to submit issues, feature requests, or improvements to enhance the application's functionality.

## 📞 Support

If you encounter any issues or need help with setup, please check the troubleshooting section above or create an issue with detailed error information.
