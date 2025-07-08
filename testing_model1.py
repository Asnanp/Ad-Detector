#!/usr/bin/env python3
"""
ULTIMATE 100% ACCURACY TEST - Danger Detection, Scam Analysis, Emotion Recognition
"""

import requests
import json
import time
from typing import List, Tuple, Dict

def test_ultimate_100_percent_accuracy():
    """Test the ultimate model for 100% accuracy with comprehensive analysis"""
    
    print("🚀 ULTIMATE 100% ACCURACY TEST - DANGER DETECTION & EMOTION ANALYSIS")
    print("=" * 100)
    
    # ULTIMATE TEST CASES FOR 100% ACCURACY
    test_cases = [
        # SAFE PERSONAL CONTENT (NO-AD)
        ("hi my name is asnan", "no_ad"),
        ("Hello, how are you today? I'm feeling great!", "no_ad"),
        ("The weather is beautiful and I'm so happy about it!", "no_ad"),
        ("I love reading books - they make me feel wonderful!", "no_ad"),
        ("My family is amazing and I'm grateful for them.", "no_ad"),
        ("Thank you for your help - I really appreciate it!", "no_ad"),
        ("The meeting went well and I'm satisfied with the outcome.", "no_ad"),
        ("I need to buy groceries for dinner tonight because we're having guests over", "no_ad"),
        ("Our company is offering a special promotion this month for employees", "no_ad"),
        ("My friend told me about this great restaurant that offers buy one get one free appetizers", "no_ad"),
        ("I'm excited about the amazing new project we're working on at the office", "no_ad"),
        ("The incredible weather today makes me want to go for a fantastic walk", "no_ad"),
        
        # REGULAR ADVERTISEMENTS (AD)
        ("Get 50% off all items in our store! Sale ends Sunday!", "ad"),
        ("free offer now buy 1 get 1 free 50% discount", "ad"),
        ("20 FREE SPINS 100% up to €500 and 60 extra FREE spins with your 1st deposit CLAIM NOW 100% safe & Secure", "ad"),
        ("Limited time offer! Order now and save big money!", "ad"),
        ("Call now for amazing deals and special offers!", "ad"),
        ("Visit our website for exclusive bargains!", "ad"),
        ("Don't wait! This offer expires soon!", "ad"),
        ("Hurry! Only while supplies last!", "ad"),
        ("Act fast! Limited stock available!", "ad"),
        ("Subscribe to our newsletter for exclusive deals!", "ad"),
        
        # EXTREME DANGER ADS (Scams, Phishing, Dangerous)
        ("URGENT: Your account has been suspended! Click this link immediately to verify your identity!", "ad"),
        ("SECURITY ALERT: Click here now to confirm your account or it will be permanently locked!", "ad"),
        ("IMMEDIATE ACTION REQUIRED: Your payment method has expired! Update payment now!", "ad"),
        ("GUARANTEED $100,000 in 30 days! Click this link for the secret method that banks don't want you to know!", "ad"),
        ("You've won $500,000 in the foreign lottery! Click here now to claim your prize before it expires!", "ad"),
        ("VIRUS ALERT: Your computer is infected! Download this antivirus now or lose all your data!", "ad"),
        ("Make easy money from home! Guaranteed $500 daily with no risk! Limited time offer!", "ad"),
        ("CASH PRIZE: You've been selected to win $10,000! Don't miss this amazing opportunity!", "ad"),
        ("INHERITANCE NOTICE: You've inherited $2 million! Click this link to claim your inheritance immediately!", "ad"),
        ("URGENT SECURITY NOTICE: Your account will be closed! Click here to confirm identity!", "ad"),
        
        # HIGH EMOTION CONTENT
        ("AMAZING DEAL! Get 50% off everything NOW! Don't miss this INCREDIBLE opportunity!", "ad"),
        ("WOW! UNBELIEVABLE savings up to 70% off! This is FANTASTIC - hurry before it's gone!", "ad"),
        ("EXCITING NEWS! Revolutionary product that will change your life FOREVER! Order TODAY!", "ad"),
        ("BREAKTHROUGH technology! AMAZING results guaranteed! You'll be THRILLED with the outcome!", "ad"),
        ("SPECTACULAR offer! MIND-BLOWING discounts! This is ABSOLUTELY the best deal ever!", "ad"),
        
        # GAMBLING/CASINO ADS
        ("INCREDIBLE casino bonus! EXCITING games! AMAZING jackpots! Join the WINNERS today!", "ad"),
        ("THRILLING casino experience! FANTASTIC bonuses! UNBELIEVABLE prizes await you!", "ad"),
        ("Experience the EXCITEMENT of Las Vegas! AMAZING slots! INCREDIBLE jackpots! Play NOW!", "ad"),
        
        # FINANCIAL SCAMS
        ("Get RICH quick! REVOLUTIONARY system! GUARANTEED success! Don't wait - start TODAY!", "ad"),
        ("URGENT: Limited spots available! Make $1000 daily! AMAZING results guaranteed!", "ad"),
        ("Start your own profitable online business! Passive income streams! Financial freedom!", "ad"),
        
        # HEALTH/BEAUTY ADS
        ("AMAZING weight loss! INCREDIBLE results in 30 days! You'll feel FANTASTIC!", "ad"),
        ("REVOLUTIONARY anti-aging! SPECTACULAR results! Look AMAZING and feel WONDERFUL!", "ad"),
        ("Transform your life! INCREDIBLE energy! AMAZING health! Feel FANTASTIC every day!", "ad"),
        
        # COMPLEX EDGE CASES
        ("Hi everyone, my name is Jennifer and I work in marketing, and I'm thrilled to be part of this amazing team that creates innovative solutions for our wonderful customers.", "no_ad"),
        ("Get an absolutely INCREDIBLE 50% discount on everything in our amazing store right now, but hurry because this FANTASTIC limited-time offer expires at midnight tonight!", "ad"),
        
        # LONG SENTENCES WITH MIXED CONTENT
        ("Hello there, how are you feeling today? I hope you're having a wonderful time and that everything is going well in your life, because I know that sometimes things can be challenging but it's important to stay positive and grateful.", "no_ad"),
        ("URGENT ALERT: Your account security has been compromised and immediate action is required! Click this link now to verify your identity and prevent permanent account suspension, or you will lose access to all your important data and financial information forever!", "ad"),
    ]
    
    print(f"📊 Testing {len(test_cases)} ultimate test cases for 100% accuracy...")
    print(f"🎯 Including danger detection, scam analysis, emotion recognition, and naturalness assessment")
    print()
    
    # Test each case
    results = []
    correct_predictions = 0
    total_predictions = len(test_cases)
    
    for i, (text, expected) in enumerate(test_cases, 1):
        print(f"Test {i:2d}/{total_predictions}: ", end="")
        
        try:
            # Make API request
            response = requests.post(
                'http://localhost:5000/predict_text',
                json={'text': text},
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get('prediction', 'unknown').lower()
                confidence = result.get('confidence', 0)
                danger_analysis = result.get('danger_analysis', {})
                scam_analysis = result.get('scam_analysis', {})
                emotion_analysis = result.get('emotion_analysis', {})
                naturalness_analysis = result.get('naturalness_analysis', {})
                
                is_correct = prediction == expected
                if is_correct:
                    correct_predictions += 1
                
                status = "✅" if is_correct else "❌"
                
                # Store detailed results
                results.append({
                    'text': text,
                    'expected': expected,
                    'predicted': prediction,
                    'confidence': confidence,
                    'correct': is_correct,
                    'length': len(text),
                    'word_count': len(text.split()),
                    'danger_analysis': danger_analysis,
                    'scam_analysis': scam_analysis,
                    'emotion_analysis': emotion_analysis,
                    'naturalness_analysis': naturalness_analysis
                })
                
                # Print result with comprehensive analysis
                print(f"{status} {prediction.upper()} ({confidence:.1%}) [Expected: {expected.upper()}]")
                if len(text) > 80:
                    print(f"     Text: \"{text[:77]}...\"")
                else:
                    print(f"     Text: \"{text}\"")
                
                # Print comprehensive analysis
                danger_level = danger_analysis.get('danger_level', 'SAFE')
                danger_score = danger_analysis.get('danger_score', 0)
                scam_type = scam_analysis.get('primary_scam_type', 'none')
                scam_confidence = scam_analysis.get('primary_scam_confidence', 0)
                emotional_state = emotion_analysis.get('emotional_state', 'NEUTRAL')
                dominant_emotions = emotion_analysis.get('dominant_emotions', [])
                naturalness_level = naturalness_analysis.get('naturalness_level', 'UNKNOWN')
                naturalness_percentage = naturalness_analysis.get('naturalness_percentage', 0)
                
                print(f"     🚨 Danger: {danger_level} (Score: {danger_score})")
                print(f"     🔍 Scam: {scam_type} ({scam_confidence:.0f}%)")
                print(f"     🎭 Emotion: {emotional_state} | Dominant: {[emotion[0] for emotion in dominant_emotions[:2]]}")
                print(f"     😊 Natural: {naturalness_level} ({naturalness_percentage:.0f}%)")
                
                if not is_correct:
                    print(f"     ⚠️  MISCLASSIFICATION DETECTED!")
                    print(f"     🔧 Debug Info: Features={result.get('features_analyzed', 0)}, Model={result.get('model_type', 'unknown')}")
                print()
                
            else:
                print(f"❌ API Error: {response.status_code}")
                results.append({
                    'text': text,
                    'expected': expected,
                    'predicted': 'error',
                    'confidence': 0,
                    'correct': False,
                    'length': len(text),
                    'word_count': len(text.split()),
                    'danger_analysis': {},
                    'scam_analysis': {},
                    'emotion_analysis': {},
                    'naturalness_analysis': {}
                })
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results.append({
                'text': text,
                'expected': expected,
                'predicted': 'error',
                'confidence': 0,
                'correct': False,
                'length': len(text),
                'word_count': len(text.split()),
                'danger_analysis': {},
                'scam_analysis': {},
                'emotion_analysis': {},
                'naturalness_analysis': {}
            })
        
        # Small delay to avoid overwhelming the server
        time.sleep(0.1)
    
    # Calculate ultimate statistics
    overall_accuracy = (correct_predictions / total_predictions) * 100
    
    # Separate by category
    ad_cases = [r for r in results if r['expected'] == 'ad']
    no_ad_cases = [r for r in results if r['expected'] == 'no_ad']
    
    ad_correct = sum(1 for r in ad_cases if r['correct'])
    no_ad_correct = sum(1 for r in no_ad_cases if r['correct'])
    
    ad_accuracy = (ad_correct / len(ad_cases)) * 100 if ad_cases else 0
    no_ad_accuracy = (no_ad_correct / len(no_ad_cases)) * 100 if no_ad_cases else 0
    
    # Danger level analysis
    extreme_danger_cases = [r for r in results if r['danger_analysis'].get('danger_level') == 'EXTREME_DANGER']
    high_danger_cases = [r for r in results if r['danger_analysis'].get('danger_level') == 'HIGH_DANGER']
    safe_cases = [r for r in results if r['danger_analysis'].get('danger_level') == 'SAFE']
    
    # Scam detection analysis
    scam_cases = [r for r in results if r['scam_analysis'].get('is_likely_scam', False)]
    
    # Average confidence and metrics
    avg_confidence = sum(r['confidence'] for r in results if r['confidence'] > 0) / len([r for r in results if r['confidence'] > 0])
    avg_danger_score = sum(r['danger_analysis'].get('danger_score', 0) for r in results) / len(results)
    
    # Print ultimate results
    print("🏆 ULTIMATE 100% ACCURACY TEST RESULTS")
    print("=" * 100)
    print(f"📊 Overall Accuracy: {overall_accuracy:.1f}% ({correct_predictions}/{total_predictions})")
    print()
    print("📈 Category Breakdown:")
    print(f"   🔴 AD Accuracy:     {ad_accuracy:.1f}% ({ad_correct}/{len(ad_cases)})")
    print(f"   🟢 NO-AD Accuracy:  {no_ad_accuracy:.1f}% ({no_ad_correct}/{len(no_ad_cases)})")
    print()
    print("🚨 Danger Detection Analysis:")
    print(f"   ⚠️  Extreme Danger Cases: {len(extreme_danger_cases)}")
    print(f"   ⚠️  High Danger Cases:    {len(high_danger_cases)}")
    print(f"   ✅ Safe Cases:           {len(safe_cases)}")
    print()
    print("🔍 Scam Detection Analysis:")
    print(f"   🚨 Likely Scams Detected: {len(scam_cases)}")
    print()
    print("📊 Ultimate Metrics:")
    print(f"   🎯 Average Confidence:     {avg_confidence:.1%}")
    print(f"   🚨 Average Danger Score:   {avg_danger_score:.1f}")
    print(f"   🔧 Features Analyzed:      150 advanced features")
    print(f"   🧠 Model Type:             Ultimate Advanced ML")
    print()
    
    # Show misclassifications with detailed analysis
    misclassifications = [r for r in results if not r['correct']]
    if misclassifications:
        print("❌ MISCLASSIFICATIONS WITH DETAILED ANALYSIS:")
        print("-" * 80)
        for i, miss in enumerate(misclassifications, 1):
            print(f"{i}. Expected: {miss['expected'].upper()}, Got: {miss['predicted'].upper()}")
            print(f"   Text: \"{miss['text'][:100]}{'...' if len(miss['text']) > 100 else ''}\"")
            print(f"   Confidence: {miss['confidence']:.1%}")
            danger = miss['danger_analysis']
            scam = miss['scam_analysis']
            emotion = miss['emotion_analysis']
            natural = miss['naturalness_analysis']
            print(f"   Danger: {danger.get('danger_level', 'UNKNOWN')} (Score: {danger.get('danger_score', 0)})")
            print(f"   Scam: {scam.get('primary_scam_type', 'none')} ({scam.get('primary_scam_confidence', 0):.0f}%)")
            print(f"   Emotion: {emotion.get('emotional_state', 'UNKNOWN')}")
            print(f"   Natural: {natural.get('naturalness_level', 'UNKNOWN')} ({natural.get('naturalness_percentage', 0):.0f}%)")
            print()
    
    # Ultimate performance assessment
    if overall_accuracy == 100.0:
        print("🎉 PERFECT! 100% ACCURACY ACHIEVED!")
        print("🚀 ULTIMATE MODEL IS PRODUCTION-READY!")
        print("🧠 DANGER DETECTION, SCAM ANALYSIS, AND EMOTION RECOGNITION WORKING PERFECTLY!")
        print("✨ ALL ADVANCED FEATURES FUNCTIONING OPTIMALLY!")
    elif overall_accuracy >= 95:
        print("🎯 EXCELLENT! Near-perfect performance achieved!")
        print("🚀 Model is ready for production with minor fine-tuning!")
    elif overall_accuracy >= 90:
        print("✅ VERY GOOD! Strong performance with room for improvement!")
    else:
        print("⚠️  NEEDS IMPROVEMENT! Significant optimization required!")
    
    return results

if __name__ == "__main__":
    test_ultimate_100_percent_accuracy()
