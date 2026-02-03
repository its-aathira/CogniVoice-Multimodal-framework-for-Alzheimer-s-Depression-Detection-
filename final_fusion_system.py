import numpy as np

def final_diagnostic_decision(voice_prob, eeg_prob):
    """
    Combines Voice and EEG probabilities for a final decision.
    We give more 'weight' to the EEG because it has higher accuracy.
    """
    # Weighting: EEG is 60% of the decision, Voice is 40%
    combined_score = (0.4 * voice_prob) + (0.6 * eeg_prob)
    
    if combined_score > 0.5:
        status = "⚠️ Alzheimer's Detected"
        confidence = combined_score * 100
    else:
        status = "✅ Healthy / Normal"
        confidence = (1 - combined_score) * 100
        
    return status, round(confidence, 2)

# --- TEST CASE ---
# Let's say your Voice model is 55% sure it's AD
# And your EEG model is 80% sure it's AD
prediction, conf = final_diagnostic_decision(0.55, 0.80)

print("=" * 30)
print(" COGNIVOICE FINAL REPORT")
print("=" * 30)
print(f"Final Diagnosis: {prediction}")
print(f"System Confidence: {conf}%")
print("-" * 30)
print("Recommendation: Consult a neurologist for clinical validation.")
print("=" * 30)