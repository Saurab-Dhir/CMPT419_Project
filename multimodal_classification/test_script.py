from multimodal_classification_model import Evidence, MultiModalClassifier

# Parameters
TONE_CLASSIFIER_RELIABILITY = 0.68
FACE_CLASSIFIER_RELIABILITY = 0.87
SEMANTIC_CLASSIFIER_RELIABILITY = 0.85


print("Running Test Cases for MultiModalClassifier\n")
tests = []
model = MultiModalClassifier()

# ======================
#      TEST CASES 
# ======================

# Description: All modalities show the same positive evidence ('happy') with high confidence and reliability.
# Purpose: Verifies basic fusion when all evidences agree.
test_basic = {
    'description': "Consistent 'happy' evidence for all modalities.",
    'tone': Evidence('happy', 0.9, TONE_CLASSIFIER_RELIABILITY),
    'face': Evidence('happy', 0.9, FACE_CLASSIFIER_RELIABILITY),
    'semantics': Evidence('happy', 0.9, SEMANTIC_CLASSIFIER_RELIABILITY),
}

# Description: One modality (tone) uses an unrecognized emotion ('excited').
# Purpose: Checks that when an emotion is not in the expected set, the mass function returns full ignorance for that modality.
test_unrecognized_emotion = {
    'description': "Tone has an unrecognized emotion ('excited').",
    'tone': Evidence('excited', 0.8, TONE_CLASSIFIER_RELIABILITY),  # 'excited' is not in emotion set.
    'face': Evidence('sad', 0.9, FACE_CLASSIFIER_RELIABILITY),
    'semantics': Evidence('happy', 0.9, SEMANTIC_CLASSIFIER_RELIABILITY)
}

# Description: The tone evidence has zero confidence.
# Purpose: Tests how a modality with no confidence (i.e. full ignorance in that channel) affects the fusion.
test_tone_zero_confidence = {
    'description': "Tone evidence with zero confidence.",
    'tone': Evidence('sad', 0.0, TONE_CLASSIFIER_RELIABILITY),
    'face': Evidence('neutral', 0.7, FACE_CLASSIFIER_RELIABILITY),
    'semantics': Evidence('neutral', 0.8, SEMANTIC_CLASSIFIER_RELIABILITY)
}

# Description: The nonverbal evidences conflict.
# Purpose: Checks that all evidences contribute to the fused prediction.
test_mixed_emotion_nonverbal = {
    'description': "Mixed emotions within nonverbal cues.",
    'tone': Evidence('disgust', 0.9, TONE_CLASSIFIER_RELIABILITY),
    'face': Evidence('happy', 0.7, FACE_CLASSIFIER_RELIABILITY),
    'semantics': Evidence('disgust', 0.8, SEMANTIC_CLASSIFIER_RELIABILITY)
}

# Description: Incongruent evidence where nonverbal cues (tone and face) are positive ('happy') 
#              but the semantics evidence is negative ('sad').
# Purpose: Verifies the incongruence discount (INCONGRUENCE_DISCOUNT) is applied to the semantics modality.
test_incongruent_pos_nonverbal = {
    'description': "Incongruent evidence (positive nonverbal vs. negative semantics).",
    'tone': Evidence('happy', 0.9, TONE_CLASSIFIER_RELIABILITY),
    'face': Evidence('happy', 0.9, FACE_CLASSIFIER_RELIABILITY),
    'semantics': Evidence('sad', 0.9, SEMANTIC_CLASSIFIER_RELIABILITY)
}

# Description: Incongruent evidence where nonverbal cues are negative (e.g., 'sad' and 'angry') 
#              while the semantic evidence is positive ('happy').
# Purpose: Tests the discounting when semantic evidence conflicts with nonverbal modalities in the opposite direction.
test_incongruent_neg_nonverbal = {
    'description': "Incongruent evidence (negative nonverbal vs. positive semantics).",
    'tone': Evidence('sad', 0.9, TONE_CLASSIFIER_RELIABILITY),
    'face': Evidence('angry', 0.7, FACE_CLASSIFIER_RELIABILITY),
    'semantics': Evidence('happy', 0.8, SEMANTIC_CLASSIFIER_RELIABILITY)
}

# Description: Uses borderline confidence values (0 and 1) for different modalities.
# Purpose: Verifies that the mass function calculation properly handles absolute confidence values.
test_edge_case_zero_confidence = {
    'description': "Edge case values where confidence is 0 for all evidences.",
    'tone': Evidence('fearful', 0, TONE_CLASSIFIER_RELIABILITY),
    'face': Evidence('sad', 0, FACE_CLASSIFIER_RELIABILITY),
    'semantics': Evidence('fearful', 0, SEMANTIC_CLASSIFIER_RELIABILITY)
}

# Description: Mixed emotions across modalities with differing confidence and reliability values.
# Purpose: Checks the model’s performance in a more complex situation with variable inputs.
test_varying_mixed_evidence = {
    'description': "Mixed evidence with varying confidence and reliability.",
    'tone': Evidence('happy', 0.7, TONE_CLASSIFIER_RELIABILITY),
    'face': Evidence('sad', 0.7, FACE_CLASSIFIER_RELIABILITY),
    'semantics': Evidence('neutral', 0.7, SEMANTIC_CLASSIFIER_RELIABILITY)
}

# Description: High conflict scenario with strongly conflicting evidence (e.g., 'happy' vs. 'sad').
# Purpose: Evaluates how the combination of mass functions behaves when there is significant conflict.
test_high_confidence_conflicting_evidence = {
    'description': "High conflict scenario with strongly confidence levels.",
    'tone': Evidence('happy', 0.95, TONE_CLASSIFIER_RELIABILITY),
    'face': Evidence('sad', 0.95, FACE_CLASSIFIER_RELIABILITY),
    'semantics': Evidence('neutral', 0.95, SEMANTIC_CLASSIFIER_RELIABILITY)
}


# Description: Provide an invalid confidence value (above 1) to trigger a ValueError.
# Purpose: Verifies that the Evidence class correctly validates input parameters.
print("Invalid confidence value (above 1) should trigger an exception.")
try:
    test_invalid_evidence = {
        'description': "Invalid confidence value (above 1) should trigger an exception.",
        'tone': Evidence('happy', 1.1, TONE_CLASSIFIER_RELIABILITY),  # Invalid confidence
        'face': Evidence('happy', 0.8, FACE_CLASSIFIER_RELIABILITY),
        'semantics': Evidence('happy', 0.9, SEMANTIC_CLASSIFIER_RELIABILITY)
    }
except ValueError as e:
    print("Error detected:", e)

print('\n\n\n')

# Basic cases
tests.extend([test_basic])

# Mixed emotions
tests.extend([
    test_mixed_emotion_nonverbal, 
    test_varying_mixed_evidence, 
    test_high_confidence_conflicting_evidence
])

# Incongruence
tests.extend([
    test_incongruent_pos_nonverbal, 
    test_incongruent_neg_nonverbal
])

# Special cases
tests.extend([
    test_unrecognized_emotion, 
    test_tone_zero_confidence, 
    test_edge_case_zero_confidence
])

for idx, test_case in enumerate(tests):
    print(f"Test case {idx + 1}: {test_case['description']}")
    tone = test_case['tone']
    face = test_case['face']
    semantics = test_case['semantics']
    betp = model.predict(tone, face, semantics)
    evidences = f"tone = {tone.emotion}, facial expression = {face.emotion}, semantics = {semantics.emotion}"
    model.print_mass_function(betp, evidences)
    print("-" * 50)