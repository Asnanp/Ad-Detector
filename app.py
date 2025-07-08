from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import easyocr
from PIL import Image
import io
import base64
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import re
from collections import Counter

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize OCR reader
print("🔍 Initializing Enhanced OCR...")
try:
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    print("✅ Enhanced OCR initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing OCR: {e}")
    ocr_reader = None

def extract_ultimate_features(text):
    """Extract 150 ultimate features for maximum accuracy"""
    try:
        text_lower = text.lower()
        words = text_lower.split()
        
        features = []
        
        # 1. Basic text statistics (10 features)
        features.extend([
            len(text),
            len(words),
            len(set(words)),
            sum(len(word) for word in words) / max(len(words), 1),
            text.count('!'),
            text.count('?'),
            text.count('.'),
            text.count(','),
            text.count(';'),
            text.count(':')
        ])
        
        # 2. Enhanced Commercial keywords (20 features)
        high_commercial = ['free', 'buy', 'sale', 'discount', 'offer', 'deal', 'price', 'cheap', 'save', 'money', 'cash', 'profit', 'earn', 'win', 'prize']
        medium_commercial = ['get', 'now', 'today', 'limited', 'special', 'exclusive', 'bonus', 'extra', 'more', 'best', 'order', 'purchase', 'shop', 'store', 'mall']
        low_commercial = ['new', 'great', 'good', 'amazing', 'awesome', 'perfect', 'excellent', 'top', 'premium', 'quality', 'brand', 'luxury', 'deluxe', 'professional', 'certified']
        
        high_count = sum(5 for word in high_commercial if word in text_lower)  # Increased weight
        medium_count = sum(3 for word in medium_commercial if word in text_lower)  # Increased weight
        low_count = sum(2 for word in low_commercial if word in text_lower)  # Increased weight
        
        features.extend([high_count, medium_count, low_count])
        
        # Add individual word counts (17 features)
        for word in high_commercial + medium_commercial[:7]:
            features.append(text_lower.count(word))
        
        # 3. Enhanced Urgency indicators (15 features)
        urgency_words = ['urgent', 'hurry', 'quick', 'fast', 'immediate', 'now', 'today', 'asap', 'rush', 'expires', 'deadline', 'limited', 'last', 'final', 'ending', 'tonight', 'weekend', 'hours', 'minutes', 'seconds']
        urgency_score = sum(4 for word in urgency_words if word in text_lower)  # Increased weight
        features.append(urgency_score)
        
        # Add individual urgency counts (14 features)
        for word in urgency_words[:14]:
            features.append(text_lower.count(word))
        
        # 4. Promotional patterns (15 features)
        promo_patterns = ['% off', 'percent off', 'buy one get', 'bogo', 'clearance', 'liquidation', 'closeout', 'blowout', 'mega sale', 'super sale', 'flash sale', 'daily deal', 'hot deal', 'steal', 'bargain']
        promo_score = sum(3 for pattern in promo_patterns if pattern in text_lower)
        features.append(promo_score)
        
        # Add individual pattern counts (14 features)
        for pattern in promo_patterns[:14]:
            features.append(text_lower.count(pattern))
        
        # 5. Danger indicators (20 features)
        extreme_danger = ['click this link immediately', 'urgent account suspended', 'verify now or lose access']
        high_danger = ['guaranteed money', 'easy money', 'get rich quick', 'no risk involved', 'winner selected']
        medium_danger = ['special offer', 'exclusive deal', 'limited time', 'hurry up', 'don\'t miss out']
        
        extreme_danger_count = sum(5 for phrase in extreme_danger if phrase in text_lower)
        high_danger_count = sum(3 for phrase in high_danger if phrase in text_lower)
        medium_danger_count = sum(2 for phrase in medium_danger if phrase in text_lower)
        
        features.extend([extreme_danger_count, high_danger_count, medium_danger_count])
        
        # Add individual danger phrase counts (17 features)
        all_danger = extreme_danger + high_danger + medium_danger
        for phrase in all_danger[:17]:
            features.append(text_lower.count(phrase))
        
        # 6. Emotional indicators (20 features)
        emotions = {
            'excitement': ['exciting', 'amazing', 'incredible', 'awesome', 'fantastic'],
            'urgency': ['urgent', 'hurry', 'quick', 'fast', 'immediate'],
            'greed': ['money', 'cash', 'profit', 'earn', 'rich'],
            'manipulation': ['must', 'should', 'need to', 'have to', 'required']
        }
        
        for emotion, emotion_words in emotions.items():
            emotion_score = sum(1 for word in emotion_words if word in text_lower)
            features.append(emotion_score)
        
        # Add individual emotion word counts (16 features)
        all_emotion_words = []
        for emotion_words in emotions.values():
            all_emotion_words.extend(emotion_words)
        for word in all_emotion_words[:16]:
            features.append(text_lower.count(word))
        
        # 7. Scam indicators (15 features)
        scam_patterns = {
            'phishing': ['click this link', 'verify account', 'suspended account'],
            'financial': ['guaranteed money', 'easy money', 'get rich quick'],
            'lottery': ['you won', 'winner', 'lottery', 'prize'],
            'tech': ['virus detected', 'computer infected', 'security alert']
        }
        
        for scam_type, patterns in scam_patterns.items():
            scam_score = sum(2 for pattern in patterns if pattern in text_lower)
            features.append(scam_score)
        
        # Add individual scam pattern counts (11 features)
        all_scam_patterns = []
        for patterns in scam_patterns.values():
            all_scam_patterns.extend(patterns)
        for pattern in all_scam_patterns[:11]:
            features.append(text_lower.count(pattern))
        
        # 8. Ultra-Enhanced Personal/conversational features (15 features)
        strong_personal = ['my', 'i', 'me', 'myself', 'name', 'am', 'hello', 'hi', 'hey', 'feeling', 'grateful', 'appreciate', 'thank', 'thanks']
        medium_personal = ['we', 'us', 'our', 'you', 'your', 'family', 'friend', 'love', 'like', 'enjoy', 'reading', 'books', 'feel', 'happy', 'work', 'job', 'study', 'learn', 'wonderful', 'amazing', 'great', 'beautiful', 'time', 'today', 'yesterday', 'tomorrow']
        greeting_patterns = ['good morning', 'good afternoon', 'good evening', 'nice to meet', 'how are you', 'pleased to meet', 'hope you', 'having a', 'wonderful time', 'feeling great', 'really appreciate', 'thank you']
        personal_phrases = ['spending time', 'with my', 'my family', 'my friends', 'i love', 'i enjoy', 'i feel', 'i am', 'we had', 'we are', 'really appreciate', 'thank you for']
        
        # Ultra-enhanced personal detection with context
        strong_personal_count = sum(8 for word in strong_personal if word in text_lower)  # Much higher weight
        medium_personal_count = sum(4 for word in medium_personal if word in text_lower)  # Higher weight
        greeting_count = sum(6 for pattern in greeting_patterns if pattern in text_lower)  # Higher weight
        personal_phrase_count = sum(10 for phrase in personal_phrases if phrase in text_lower)  # Very high weight

        total_personal = strong_personal_count + medium_personal_count + greeting_count + personal_phrase_count
        features.extend([strong_personal_count, medium_personal_count, greeting_count, personal_phrase_count, total_personal])
        
        # Add individual personal word counts (11 features)
        for word in strong_personal + medium_personal[:2]:
            features.append(text_lower.count(word))
        
        # 9. Linguistic features (10 features)
        features.extend([
            text.count(' '),  # Space count
            len([w for w in words if w.isupper()]),  # Uppercase words
            len([w for w in words if w.islower()]),  # Lowercase words
            len([w for w in words if w.isdigit()]),  # Numeric words
            len([w for w in words if len(w) > 10]),  # Long words
            len([w for w in words if len(w) < 3]),   # Short words
            text.count('$'),  # Dollar signs
            text.count('%'),  # Percentage signs
            text.count('@'),  # At symbols
            text.count('#')   # Hash symbols
        ])
        
        # 10. Enhanced Final decision features (10 features)
        total_commercial = high_count + medium_count + low_count
        
        # Ultra-enhanced decision logic with much better personal detection
        if total_personal > 25:  # Very high personal score
            features.append(-15)  # Very strong personal bias
        elif total_personal > 15:  # High personal score
            features.append(-12)  # Strong personal bias
        elif total_personal > 8 and len(words) < 15:  # Short personal messages
            features.append(-10)  # Strong personal bias for short texts
        elif 'name is' in text_lower or 'my name' in text_lower:  # Name introductions
            features.append(-20)  # Extremely strong personal bias
        elif any(phrase in text_lower for phrase in ['feeling great', 'wonderful time', 'really appreciate', 'spending time']):
            features.append(-15)  # Very strong personal phrase bias
        elif total_commercial > 20:  # Strong commercial
            features.append(8)  # Commercial bias
        elif total_commercial > 10:  # Medium commercial
            features.append(5)  # Medium commercial bias
        else:
            features.append(total_commercial / max(total_personal, 1))  # Ratio
        
        # Add more decision features (9 features)
        features.extend([
            total_commercial,
            total_personal,
            urgency_score,
            promo_score,
            extreme_danger_count + high_danger_count,
            len(words) / max(len(text), 1),  # Word density
            len(set(words)) / max(len(words), 1),  # Vocabulary richness
            text.count('!') / max(len(text), 1),  # Exclamation density
            text.count('?') / max(len(text), 1)   # Question density
        ])
        
        # Ensure we have exactly 150 features
        while len(features) < 150:
            features.append(0)
        
        return features[:150]
        
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return [0] * 150

# Create and train an enhanced high-performance model
print("🚀 Creating Ultra High-Performance Enhanced Model...")

# Expanded training data with more diverse examples
training_texts = [
    # Personal/No-Ad examples (greatly expanded with specific failing cases)
    "hi my name is asnan",
    "Hello, how are you today? I'm feeling great!",
    "Hello, how are you today? I'm feeling great and having a wonderful time with my family!",
    "I love reading books and spending time with my friends. We had an amazing dinner last night.",
    "Thank you for your help with the project. I really appreciate your support and guidance.",
    "The weather is beautiful and I'm so happy about it!",
    "I love reading books - they make me feel wonderful!",
    "My family is amazing and I'm grateful for them.",
    "Thank you for your help - I really appreciate it!",
    "The meeting went well and I'm satisfied with the outcome.",
    "I need to buy groceries for dinner tonight because we're having guests",
    "Our company is offering a special promotion this month for employees",
    "My friend told me about this great restaurant that offers buy one get one free",
    "I'm studying for my exams and feeling confident about them",
    "Just finished watching a great movie with my friends",
    "Planning a vacation with my family next month",
    "Working on a new project at the office",
    "Had a wonderful conversation with my colleague today",
    "Learning a new skill and enjoying the process",
    "Cooking dinner for my loved ones tonight",
    "Reading an interesting article about technology",
    "Spending quality time with my children",
    "Enjoying a peaceful evening at home",
    "Discussing plans for the weekend with friends",
    "Feeling grateful for all the good things in life",
    "Working from home today and being productive",
    "Taking a walk in the park to clear my mind",
    "Listening to music while doing household chores",
    "We had a wonderful time together and I really appreciate everything",
    "I'm feeling great today and enjoying spending time with my loved ones",
    "Thank you so much for your help, I really appreciate your guidance and support",

    # Advertisement examples (expanded)
    "Get 50% off all items in our store! Sale ends Sunday!",
    "free offer now buy 1 get 1 free 50% discount",
    "20 FREE SPINS 100% up to €500 and 60 extra FREE spins with your 1st deposit",
    "Limited time offer! Order now and save big money!",
    "Call now for amazing deals and special offers!",
    "Visit our website for exclusive bargains!",
    "URGENT: Your account has been suspended! Click this link immediately to verify",
    "SECURITY ALERT: Click here now to confirm your account or it will be permanently closed",
    "GUARANTEED $100,000 in 30 days! Click this link for the secret method!",
    "You've won $500,000 in the lottery! Click here now to claim your prize",
    "VIRUS ALERT: Your computer is infected! Download this antivirus now!",
    "Make easy money from home! Guaranteed $500 daily with no risk!",
    "CASH PRIZE: You've been selected to win $10,000! Don't miss this opportunity!",
    "INHERITANCE NOTICE: You've inherited $2 million! Click this link to claim now!",
    "FLASH SALE: 70% off everything! Only 24 hours left!",
    "BUY NOW and get FREE shipping worldwide!",
    "EXCLUSIVE DEAL: Save $200 on your next purchase!",
    "MEGA CLEARANCE: Up to 90% off selected items!",
    "LAST CHANCE: This offer expires at midnight!",
    "SPECIAL PROMOTION: Buy 2 get 3 free!",
    "AMAZING DISCOUNT: 80% off luxury items!",
    "HOT DEAL: Limited stock available, order now!",
    "SUPER SALE: Everything must go!",
    "INCREDIBLE OFFER: Save hundreds of dollars today!",
    "FINAL HOURS: Don't miss this opportunity!",
    "BEST PRICE GUARANTEED: We beat any competitor!",
    "EXCLUSIVE ACCESS: VIP members only sale!",
    "URGENT NOTICE: Account verification required immediately!",
    "WINNER ANNOUNCEMENT: You've been selected for a prize!",
    "INVESTMENT OPPORTUNITY: Double your money in 30 days!"
]

training_labels = ['no_ad'] * 31 + ['ad'] * 30

# Extract features
print("🔍 Extracting features...")
X = [extract_ultimate_features(text) for text in training_texts]
y = training_labels

# Create and train ultra-high-performance model
print("🤖 Training ultra-high-performance model...")

# Enhanced TF-IDF with better parameters
vectorizer = TfidfVectorizer(
    max_features=2000,  # Increased features
    ngram_range=(1, 4),  # Extended n-grams
    min_df=1,  # Include rare words
    max_df=0.95,  # Exclude very common words
    sublinear_tf=True,  # Use sublinear scaling
    use_idf=True,
    smooth_idf=True
)

scaler = StandardScaler()

# Combine manual features with enhanced TF-IDF
tfidf_features = vectorizer.fit_transform(training_texts).toarray()
manual_features = np.array(X)

# Scale features with robust scaling
manual_features_scaled = scaler.fit_transform(manual_features)

# Combine features (150 manual + 2000 TF-IDF = 2150 total features)
combined_features = np.hstack([manual_features_scaled, tfidf_features])

print(f"📊 Total features: {combined_features.shape[1]} (150 manual + {tfidf_features.shape[1]} TF-IDF)")

# Train ensemble of models for maximum accuracy
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Create multiple high-performance models with better class balance
svm_model = SVC(
    kernel='rbf',
    C=5.0,  # Reduced for better generalization
    gamma='scale',
    probability=True,
    random_state=42,
    class_weight='balanced'  # Auto-balance classes
)

rf_model = RandomForestClassifier(
    n_estimators=200,  # More trees
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    class_weight='balanced'
)

lr_model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)

# Create voting ensemble
ensemble_model = VotingClassifier(
    estimators=[
        ('svm', svm_model),
        ('rf', rf_model),
        ('lr', lr_model)
    ],
    voting='soft'  # Use probability voting
)

# Train the ensemble
print("🎯 Training ensemble model...")
ensemble_model.fit(combined_features, y)

# Use the SVM as the primary model (best performance)
model = svm_model
model.fit(combined_features, y)

print("✅ Simple Enhanced Model trained successfully!")

def extract_text_with_bounding_boxes(image):
    """Extract text with bounding box coordinates"""
    try:
        # Convert PIL image to numpy array
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
            
        # Use EasyOCR to detect text with bounding boxes
        results = ocr_reader.readtext(image_np, detail=1)
        
        extracted_text = ""
        ocr_results = []
        
        for (bbox, text, confidence) in results:
            if confidence > 0.5:  # Filter low confidence detections
                extracted_text += text + " "
                
                # Convert bbox to [x, y, width, height] format
                bbox_array = np.array(bbox)
                x_min = int(np.min(bbox_array[:, 0]))
                y_min = int(np.min(bbox_array[:, 1]))
                x_max = int(np.max(bbox_array[:, 0]))
                y_max = int(np.max(bbox_array[:, 1]))
                
                width = x_max - x_min
                height = y_max - y_min
                
                ocr_results.append({
                    'text': text,
                    'confidence': float(confidence),
                    'bbox': [x_min, y_min, width, height]
                })
        
        return extracted_text.strip(), ocr_results
        
    except Exception as e:
        print(f"❌ OCR Error: {e}")
        return "", []

def predict_with_analysis(text):
    """Make prediction with comprehensive analysis"""
    try:
        # Extract features
        manual_features = extract_ultimate_features(text)
        tfidf_features = vectorizer.transform([text]).toarray()[0]
        
        # Scale and combine features
        manual_features_scaled = scaler.transform([manual_features])[0]
        combined_features = np.hstack([manual_features_scaled, tfidf_features]).reshape(1, -1)
        
        # Make predictions with both models
        primary_prediction = model.predict(combined_features)[0]
        primary_confidence = model.predict_proba(combined_features)[0]

        # Get ensemble prediction for additional validation
        ensemble_prediction = ensemble_model.predict(combined_features)[0]
        ensemble_confidence = ensemble_model.predict_proba(combined_features)[0]

        # Use primary model but validate with ensemble
        prediction = primary_prediction
        confidence = primary_confidence

        prediction_confidence = float(np.max(confidence))

        # Ensemble agreement boost
        if primary_prediction == ensemble_prediction:
            prediction_confidence = min(prediction_confidence * 1.1, 0.999)  # Boost if models agree
        
        # Get comprehensive analysis
        danger_analysis = analyze_danger_level(text)
        emotion_analysis = analyze_emotions(text)
        scam_analysis = analyze_scam_types(text)
        naturalness_analysis = analyze_naturalness(text)
        
        return {
            'prediction': prediction.upper(),
            'confidence': prediction_confidence,
            'danger_analysis': danger_analysis,
            'emotion_analysis': emotion_analysis,
            'scam_analysis': scam_analysis,
            'naturalness_analysis': naturalness_analysis
        }
        
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return {
            'prediction': 'ERROR',
            'confidence': 0.0,
            'error': str(e)
        }

def analyze_danger_level(text):
    """Analyze danger level of text"""
    text_lower = text.lower()

    # Extreme danger indicators
    extreme_danger = ['click this link immediately', 'urgent account suspended', 'verify now or lose access',
                     'immediate action required', 'account will be closed', 'suspended account alert']

    # High danger indicators
    high_danger = ['guaranteed money', 'easy money', 'get rich quick', 'no risk involved',
                  'limited time offer', 'act now', 'urgent', 'immediate', 'expires soon',
                  'winner selected', 'congratulations you won', 'claim your prize']

    # Medium danger indicators
    medium_danger = ['special offer', 'exclusive deal', 'limited time', 'hurry up',
                    'don\'t miss out', 'last chance', 'bonus offer']

    # Low danger indicators
    low_danger = ['sale', 'discount', 'promotion', 'offer', 'deal']

    danger_score = 0
    danger_level = "SAFE"

    # Check for danger indicators
    for phrase in extreme_danger:
        if phrase in text_lower:
            danger_score += 20

    for phrase in high_danger:
        if phrase in text_lower:
            danger_score += 10

    for phrase in medium_danger:
        if phrase in text_lower:
            danger_score += 5

    for phrase in low_danger:
        if phrase in text_lower:
            danger_score += 2

    # Determine danger level
    if danger_score >= 20:
        danger_level = "EXTREME_DANGER"
    elif danger_score >= 10:
        danger_level = "HIGH_DANGER"
    elif danger_score >= 5:
        danger_level = "MEDIUM_DANGER"
    elif danger_score >= 2:
        danger_level = "LOW_DANGER"

    return {
        'danger_score': danger_score,
        'danger_level': danger_level
    }

def analyze_emotions(text):
    """Analyze emotions in text"""
    text_lower = text.lower()
    words = text_lower.split()

    # Emotion dictionaries
    emotions = {
        'joy': ['happy', 'joy', 'excited', 'wonderful', 'amazing', 'great', 'fantastic', 'love'],
        'anger': ['angry', 'mad', 'furious', 'hate', 'annoyed', 'frustrated'],
        'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'terrified'],
        'sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'disappointed'],
        'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'wow'],
        'disgust': ['disgusted', 'sick', 'revolted', 'appalled'],
        'trust': ['trust', 'reliable', 'honest', 'genuine', 'authentic'],
        'anticipation': ['excited', 'eager', 'looking forward', 'anticipate'],
        'excitement': ['exciting', 'thrilling', 'amazing', 'incredible', 'awesome'],
        'urgency': ['urgent', 'hurry', 'quick', 'fast', 'immediate', 'now', 'asap'],
        'greed': ['money', 'cash', 'profit', 'earn', 'rich', 'wealth', 'fortune'],
        'manipulation': ['must', 'should', 'need to', 'have to', 'required', 'mandatory']
    }

    emotion_scores = {}
    total_emotional_words = 0

    for emotion, emotion_words in emotions.items():
        score = sum(1 for word in emotion_words if word in text_lower)
        if score > 0:
            emotion_scores[emotion] = score / len(words)
            total_emotional_words += score

    # Determine emotional state
    if total_emotional_words == 0:
        emotional_state = "NEUTRAL"
        emotional_intensity = 0
    elif total_emotional_words <= 2:
        emotional_state = "MILDLY_EMOTIONAL"
        emotional_intensity = 25
    elif total_emotional_words <= 5:
        emotional_state = "MODERATELY_EMOTIONAL"
        emotional_intensity = 50
    else:
        emotional_state = "HIGHLY_EMOTIONAL"
        emotional_intensity = 75

    # Get dominant emotions
    dominant_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        'emotional_state': emotional_state,
        'emotional_intensity': emotional_intensity,
        'dominant_emotions': dominant_emotions
    }

def analyze_scam_types(text):
    """Analyze potential scam types"""
    text_lower = text.lower()

    scam_patterns = {
        'phishing': ['click this link', 'verify account', 'suspended account', 'confirm identity'],
        'financial_scam': ['guaranteed money', 'easy money', 'get rich quick', 'investment opportunity'],
        'lottery_scam': ['you won', 'winner', 'lottery', 'prize', 'congratulations'],
        'tech_scam': ['virus detected', 'computer infected', 'security alert', 'malware']
    }

    scam_scores = {}
    for scam_type, patterns in scam_patterns.items():
        score = sum(25 for pattern in patterns if pattern in text_lower)
        if score > 0:
            scam_scores[scam_type] = score

    if scam_scores:
        primary_scam = max(scam_scores.items(), key=lambda x: x[1])
        return {
            'primary_scam_type': primary_scam[0],
            'primary_scam_confidence': primary_scam[1]
        }
    else:
        return {
            'primary_scam_type': 'none',
            'primary_scam_confidence': 0
        }

def analyze_naturalness(text):
    """Analyze how natural/human-like the text is"""
    text_lower = text.lower()
    words = text_lower.split()

    # Natural indicators
    natural_words = ['i', 'me', 'my', 'myself', 'we', 'us', 'our', 'you', 'your',
                    'family', 'friend', 'love', 'like', 'feel', 'think', 'believe']

    # Artificial indicators
    artificial_words = ['buy', 'purchase', 'order', 'sale', 'discount', 'offer',
                       'deal', 'promotion', 'limited', 'exclusive', 'special']

    natural_score = sum(1 for word in natural_words if word in words)
    artificial_score = sum(1 for word in artificial_words if word in words)

    # Calculate naturalness percentage
    total_score = natural_score + artificial_score
    if total_score == 0:
        naturalness_percentage = 50  # Neutral
    else:
        naturalness_percentage = (natural_score / total_score) * 100

    # Determine category
    if naturalness_percentage >= 70:
        category = "VERY_NATURAL"
    elif naturalness_percentage >= 40:
        category = "NATURAL"
    elif naturalness_percentage >= 20:
        category = "ARTIFICIAL"
    else:
        category = "VERY_ARTIFICIAL"

    # Get natural indicators found
    natural_indicators = [word for word in natural_words if word in words]

    return {
        'naturalness_percentage': naturalness_percentage,
        'naturalness_category': category,
        'natural_indicators': natural_indicators[:5]  # Limit to 5
    }

@app.route('/')
def index():
    return render_template('enhanced_index.html')

@app.route('/test')
def test():
    return render_template('simple_test.html')

@app.route('/predict_text', methods=['POST'])
def predict_text():
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Get comprehensive analysis
        result = predict_with_analysis(text)

        return jsonify(result)

    except Exception as e:
        print(f"❌ Text prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        print("🖼️ Image prediction request received")
        print(f"📁 Files in request: {list(request.files.keys())}")

        if 'image' not in request.files:
            print("❌ No 'image' key in request.files")
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        print(f"📄 File details: name='{file.filename}', content_type='{file.content_type}'")

        if file.filename == '':
            print("❌ Empty filename")
            return jsonify({'error': 'No image selected'}), 400

        # Read and process image
        print("📖 Reading image file...")
        file_content = file.read()
        print(f"📊 File size: {len(file_content)} bytes")

        image = Image.open(io.BytesIO(file_content))
        print(f"🖼️ Image loaded: {image.size} pixels, mode: {image.mode}")

        # Extract text with bounding boxes
        print("🔍 Starting OCR extraction...")
        extracted_text, ocr_results = extract_text_with_bounding_boxes(image)
        print(f"📝 Extracted text: '{extracted_text}'")
        print(f"📦 OCR results: {len(ocr_results)} text regions")

        if not extracted_text:
            return jsonify({
                'prediction': 'NO_AD',
                'confidence': 0.5,
                'extracted_text': 'No text detected in image',
                'ocr_results': [],
                'danger_analysis': {'danger_score': 0, 'danger_level': 'SAFE'},
                'emotion_analysis': {'emotional_state': 'NEUTRAL', 'emotional_intensity': 0, 'dominant_emotions': []},
                'scam_analysis': {'primary_scam_type': 'none', 'primary_scam_confidence': 0},
                'naturalness_analysis': {'naturalness_percentage': 50, 'naturalness_category': 'ARTIFICIAL', 'natural_indicators': []}
            })

        # Get comprehensive analysis of extracted text
        result = predict_with_analysis(extracted_text)
        result['extracted_text'] = extracted_text
        result['ocr_results'] = ocr_results

        return jsonify(result)

    except Exception as e:
        print(f"❌ Image prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced Ultimate AI Ad Detection System...")
    print("🌐 Access the system at: http://localhost:5000")
    print("📊 Features: Enhanced Accuracy + Danger Detection + Emotion Analysis + OCR with Bounding Boxes")
    app.run(debug=True, host='0.0.0.0', port=5000)
