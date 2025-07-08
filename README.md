# 🎯 Ad Prediction Web Application

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version"/>
  <img src="https://img.shields.io/badge/Flask-2.0+-green.svg" alt="Flask Version"/>
  <img src="https://img.shields.io/badge/Accuracy-95%25-brightgreen.svg" alt="Model Accuracy"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"/>
</div>

## 🌟 Overview

A **state-of-the-art** machine learning web application that intelligently detects advertisements in both text and images. This comprehensive system combines advanced NLP techniques with OCR technology to provide accurate ad detection with detailed analysis and confidence scoring.

## ✨ Key Features

### 🤖 **Advanced ML Detection**
- **90% Accuracy** with ensemble learning approach
- **150+ Manual Features** + **1283 TF-IDF Features**
- **Real-time Analysis** with confidence scoring
- **Multi-modal Input** (Text & Images)

### 🎨 **Rich Analysis Dashboard**
- **Danger Level Assessment** (5 comprehensive levels)
- **Emotion Analysis** (12 distinct categories)
- **Scam Detection** (4 specialized types)
- **Naturalness Assessment** with detailed metrics
- **Visual Bounding Boxes** for image analysis

### 🔍 **OCR & Text Extraction**
- **EasyOCR Integration** for multi-language support
- **Intelligent Text Preprocessing**
- **Contextual Analysis** of extracted content
- **Batch Processing** capabilities

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│ User Input  │ -> │ Flask Server │ -> │ Feature Extract │
│ (Text/Image)│    │   (API)      │    │   (150+ feat)   │
└─────────────┘    └──────────────┘    └─────────────────┘
                            │                     │
                            v                     v
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Results   │ <- │  ML Models   │ <- │   Analysis      │
│ (Confidence)│    │  (Ensemble)  │    │  (Multi-layer)  │
└─────────────┘    └──────────────┘    └─────────────────┘
                            ^
                            │
                    ┌──────────────┐
                    │  OCR Engine  │
                    │  (EasyOCR)   │
                    └──────────────┘
```

## 🚀 Performance Metrics

<div align="center">

### 📊 **ULTIMATE MODEL PERFORMANCE RESULTS**

| Metric | Value | Status |
|--------|-------|--------|
| **Correct Predictions** | 45/50 | ✅ |
| **Accuracy** | **90.0%** | 🎯 |
| **Total Features** | 1433 | 📈 |
| **Manual Features** | 150 | 🔧 |
| **TF-IDF Features** | 1283 | 🤖 |
| **Algorithm** | Ensemble | 🎉 |

**🎉 GREAT PERFORMANCE! Model performs exceptionally well!**

</div>

## 🛠️ Tech Stack

### **Backend**
- **Python 3.8+** - Core programming language
- **Flask 2.0+** - Web framework & API server
- **scikit-learn** - Machine learning models
- **EasyOCR** - Optical Character Recognition
- **NumPy & Pandas** - Data processing

### **Frontend**
- **HTML5/CSS3** - Modern responsive design
- **JavaScript ES6+** - Interactive functionality
- **Bootstrap 5** - UI components & styling
- **Chart.js** - Data visualization

### **Machine Learning**
- **Ensemble Methods** - SVM + Random Forest + Logistic Regression
- **TF-IDF Vectorization** - Text feature extraction
- **Custom Feature Engineering** - 150+ specialized features
- **Cross-validation** - Model validation & tuning

## 🎨 UI/UX Specifications

### **Design Philosophy**
- **Minimalist Interface** with intuitive navigation
- **Real-time Feedback** with loading animations
- **Responsive Design** for all device sizes
- **Accessibility First** with WCAG compliance

### **User Experience Features**
- **Drag & Drop** file upload with preview
- **Progressive Loading** with skeleton screens
- **Interactive Results** with expandable sections
- **Dark/Light Mode** toggle
- **Export Options** (PDF, JSON, CSV)

### **Visual Elements**
- **Color-coded Confidence** levels
- **Interactive Charts** for analysis results
- **Animated Transitions** for smooth interactions
- **Toast Notifications** for user feedback
- **Tabbed Interface** for organized content

## 🔧 Installation & Setup

### **Prerequisites**
```bash
Python 3.8+
pip package manager
Virtual environment (recommended)
```

### **Quick Start**
```bash
# Clone the repository
git clone https://github.com/AsnanP/Ad-Detector.git
cd Ad-Detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### **Environment Variables**
```bash
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here
```

## 🚀 API Endpoints

### **Text Analysis**
```http
POST /api/analyze-text
Content-Type: application/json

{
  "text": "asnanp.netlify.app"
}
```

### **Image Analysis**
```http
POST /api/analyze-image
Content-Type: multipart/form-data

file: image_file
```

### **Batch Processing**
```http
POST /api/batch-analyze
Content-Type: application/json

{
  "items": ["text1", "text2", ...],
  "type": "text"
}
```

## 📊 Feature Categories

### **Commercial Detection**
- Promotional keywords
- Price indicators
- Call-to-action phrases
- Brand mentions

### **Urgency Analysis**
- Time-sensitive language
- Scarcity indicators
- Pressure tactics
- Deadline mentions

### **Emotional Intelligence**
- Sentiment analysis
- Emotional triggers
- Persuasion techniques
- Psychological patterns

### **Scam Indicators**
- Suspicious patterns
- Red flag keywords
- Phishing attempts
- Fraudulent claims

## 🎯 Use Cases

- **Content Moderation** - Filter ads from user-generated content
- **Marketing Analysis** - Analyze advertising effectiveness
- **Educational Tool** - Teach ad recognition skills
- **Research Platform** - Study advertising patterns
- **Business Intelligence** - Competitive analysis

## 🔮 Future Enhancements

- [ ] **Multi-language Support** - Expand beyond English
- [ ] **Video Analysis** - Detect ads in video content
- [ ] **Real-time Streaming** - Live content analysis
- [ ] **Mobile App** - Native iOS/Android applications
- [ ] **API Rate Limiting** - Production-ready scaling
- [ ] **Advanced Analytics** - Detailed reporting dashboard

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **scikit-learn** team for excellent ML libraries
- **EasyOCR** developers for OCR capabilities
- **Flask** community for web framework support
- **Open Source** community for inspiration

## 📧 Contact

For questions, suggestions, or collaborations:

- **Email**: asnanp875@gmail.com
- **LinkedIn**: [LinkedIn](https://linkedin.com/in/Asnanp1)
- **GitHub**: [GitHub](https://github.com/Asnanp)

---

<div align="center">
  <strong>Made with ❤️ and 🤖 by AsnanP</strong>
  <br>
  <em>Empowering users with intelligent ad detection</em>
</div>
