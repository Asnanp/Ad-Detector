#!/usr/bin/env python3
"""
Test script to verify the enhanced model performance
"""

import requests
import json
import time

def test_enhanced_model():
    """Test the enhanced model with various examples"""
    
    print("🧪 TESTING ENHANCED MODEL PERFORMANCE")
    print("=" * 60)
    
    # Test cases with expected results
    test_cases = [
        # Personal/No-Ad cases
        {
            "text": "hi my name is asnan",
            "expected": "NO_AD",
            "description": "Personal introduction"
        },
        {
            "text": "Hello, how are you today? I'm feeling great and having a wonderful time with my family!",
            "expected": "NO_AD", 
            "description": "Personal conversation"
        },
        {
            "text": "I love reading books and spending time with my friends. We had an amazing dinner last night.",
            "expected": "NO_AD",
            "description": "Personal sharing"
        },
        {
            "text": "Working from home today and being very productive. The weather is beautiful outside.",
            "expected": "NO_AD",
            "description": "Personal update"
        },
        {
            "text": "Thank you for your help with the project. I really appreciate your support and guidance.",
            "expected": "NO_AD",
            "description": "Gratitude message"
        },
        
        # Advertisement cases
        {
            "text": "Get 50% off all items! Limited time offer - buy now and save big money!",
            "expected": "AD",
            "description": "Clear advertisement"
        },
        {
            "text": "FREE OFFER! Buy 1 get 1 free! Special discount ends tonight!",
            "expected": "AD",
            "description": "Promotional offer"
        },
        {
            "text": "FLASH SALE: 70% off everything! Only 24 hours left! Don't miss this incredible deal!",
            "expected": "AD",
            "description": "Flash sale ad"
        },
        {
            "text": "EXCLUSIVE DEAL: Save $200 on your next purchase! VIP members only!",
            "expected": "AD",
            "description": "Exclusive offer"
        },
        {
            "text": "MEGA CLEARANCE: Up to 90% off selected items! Everything must go!",
            "expected": "AD",
            "description": "Clearance sale"
        },
        
        # Dangerous/Scam cases
        {
            "text": "URGENT: Your account has been suspended! Click this link immediately to verify or lose access forever!",
            "expected": "AD",
            "description": "Phishing scam"
        },
        {
            "text": "SECURITY ALERT: Your computer is infected with a virus! Download this antivirus now!",
            "expected": "AD",
            "description": "Tech scam"
        },
        {
            "text": "GUARANTEED $100,000 in 30 days! Click this link for the secret method! No risk involved!",
            "expected": "AD",
            "description": "Financial scam"
        },
        {
            "text": "WINNER ANNOUNCEMENT: You've been selected to win $500,000 in the lottery! Claim your prize now!",
            "expected": "AD",
            "description": "Lottery scam"
        },
        {
            "text": "INHERITANCE NOTICE: You've inherited $2 million! Click this link to claim now!",
            "expected": "AD",
            "description": "Inheritance scam"
        },
        
        # Edge cases
        {
            "text": "Our company is offering a special promotion this month for employees",
            "expected": "NO_AD",
            "description": "Internal company communication"
        },
        {
            "text": "My friend told me about this great restaurant that offers buy one get one free",
            "expected": "NO_AD",
            "description": "Personal recommendation"
        },
        {
            "text": "INVESTMENT OPPORTUNITY: Double your money in 30 days! Guaranteed returns!",
            "expected": "AD",
            "description": "Investment scam"
        }
    ]
    
    url = "http://localhost:5000/predict_text"
    
    correct_predictions = 0
    total_tests = len(test_cases)
    results = []
    
    print(f"🎯 Running {total_tests} test cases...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            # Make request
            response = requests.post(url, json={"text": test_case["text"]})
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction", "").upper()
                confidence = result.get("confidence", 0)
                danger_level = result.get("danger_level", "UNKNOWN")
                
                # Check if prediction is correct
                is_correct = prediction == test_case["expected"]
                if is_correct:
                    correct_predictions += 1
                
                # Store result
                results.append({
                    "test_case": i,
                    "text": test_case["text"][:50] + "..." if len(test_case["text"]) > 50 else test_case["text"],
                    "description": test_case["description"],
                    "expected": test_case["expected"],
                    "predicted": prediction,
                    "confidence": confidence,
                    "danger_level": danger_level,
                    "correct": is_correct
                })
                
                # Print result
                status = "✅" if is_correct else "❌"
                print(f"{status} Test {i:2d}: {test_case['description']}")
                print(f"    Expected: {test_case['expected']}, Got: {prediction} ({confidence:.1f}%)")
                print(f"    Danger: {danger_level}")
                print()
                
            else:
                print(f"❌ Test {i}: Request failed with status {response.status_code}")
                results.append({
                    "test_case": i,
                    "description": test_case["description"],
                    "error": f"HTTP {response.status_code}"
                })
            
            # Small delay to avoid overwhelming the server
            time.sleep(0.1)
            
        except Exception as e:
            print(f"❌ Test {i}: Error - {e}")
            results.append({
                "test_case": i,
                "description": test_case["description"],
                "error": str(e)
            })
    
    # Calculate accuracy
    accuracy = (correct_predictions / total_tests) * 100
    
    print("=" * 60)
    print("📊 ENHANCED MODEL PERFORMANCE RESULTS")
    print("=" * 60)
    print(f"✅ Correct predictions: {correct_predictions}/{total_tests}")
    print(f"🎯 Accuracy: {accuracy:.1f}%")
    print(f"📈 Model features: 1433 total (150 manual + 1283 TF-IDF)")
    print(f"🤖 Algorithm: Ensemble (SVM + Random Forest + Logistic Regression)")
    
    # Performance analysis
    if accuracy >= 95:
        print(f"🏆 EXCELLENT PERFORMANCE! Model exceeds expectations!")
    elif accuracy >= 90:
        print(f"🎉 GREAT PERFORMANCE! Model performs very well!")
    elif accuracy >= 85:
        print(f"👍 GOOD PERFORMANCE! Model performs well!")
    elif accuracy >= 80:
        print(f"⚠️  FAIR PERFORMANCE! Model needs improvement!")
    else:
        print(f"❌ POOR PERFORMANCE! Model needs significant improvement!")
    
    # Detailed breakdown
    print(f"\n📋 DETAILED BREAKDOWN:")
    personal_correct = sum(1 for r in results if r.get("expected") == "NO_AD" and r.get("correct"))
    personal_total = sum(1 for r in results if r.get("expected") == "NO_AD")
    ad_correct = sum(1 for r in results if r.get("expected") == "AD" and r.get("correct"))
    ad_total = sum(1 for r in results if r.get("expected") == "AD")
    
    if personal_total > 0:
        personal_accuracy = (personal_correct / personal_total) * 100
        print(f"   👤 Personal/No-Ad: {personal_correct}/{personal_total} ({personal_accuracy:.1f}%)")
    
    if ad_total > 0:
        ad_accuracy = (ad_correct / ad_total) * 100
        print(f"   📢 Advertisement: {ad_correct}/{ad_total} ({ad_accuracy:.1f}%)")
    
    # Show failed cases
    failed_cases = [r for r in results if not r.get("correct", False)]
    if failed_cases:
        print(f"\n❌ FAILED CASES ({len(failed_cases)}):")
        for case in failed_cases:
            print(f"   • {case['description']}: Expected {case.get('expected', 'N/A')}, Got {case.get('predicted', 'ERROR')}")
    
    print(f"\n🎯 ENHANCEMENT SUMMARY:")
    print(f"   • Expanded training data: 55 examples (25 personal + 30 ads)")
    print(f"   • Enhanced features: 150 manual + 2000 TF-IDF features")
    print(f"   • Advanced algorithms: Ensemble voting classifier")
    print(f"   • Improved weights: Higher penalties for commercial keywords")
    print(f"   • Better detection: Enhanced personal message recognition")
    print(f"   • Ensemble validation: Multiple models for accuracy boost")
    
    return accuracy, results

if __name__ == "__main__":
    try:
        accuracy, results = test_enhanced_model()
        
        print(f"\n🚀 FINAL RESULT: {accuracy:.1f}% accuracy achieved!")
        
        if accuracy >= 95:
            print("🏆 MISSION ACCOMPLISHED! Ultra-high-performance model ready for production!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Make sure the Flask app is running on http://localhost:5000")
