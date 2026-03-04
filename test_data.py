import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'campusbot.settings')
django.setup()

from chatbot.hybrid_predictor import HybridChatbotPredictor
from chatbot.models import Intent, TrainingData

print("="*80)
print("CHATBOT DEBUG REPORT")
print("="*80)

# Initialize predictor
predictor = HybridChatbotPredictor()

# Check if model is loaded
if not predictor.model_data:
    print("\n❌ ERROR: No model loaded!")
    print("Please train a model first via the admin dashboard")
    exit()

print("\n✓ Model loaded successfully")
print(f"Model contains: {list(predictor.model_data.keys())}")

# Show all intents and their training data
print("\n" + "="*80)
print("TRAINING DATA IN DATABASE")
print("="*80)

intents = Intent.objects.all()
for intent in intents:
    training_data = TrainingData.objects.filter(intent=intent, is_active=True).first()
    if training_data:
        questions = training_data.get_questions()
        print(f"\n📌 Intent: {intent.name}")
        print(f"   Description: {intent.description}")
        print(f"   Questions ({len(questions)}):")
        for i, q in enumerate(questions[:3], 1):  # Show first 3
            print(f"      {i}. {q}")
        if len(questions) > 3:
            print(f"      ... and {len(questions)-3} more")
        print(f"   Answer: {training_data.answer[:100]}...")
    else:
        print(f"\n⚠️  Intent: {intent.name} - NO TRAINING DATA")

# Test predictions
print("\n" + "="*80)
print("TESTING PREDICTIONS")
print("="*80)

test_cases = [
    "What are the library hours?",
    "How do I register for classes?",
    "Where is room 401?",
    "When is tuition due?",
    "asdfghjkl",  # gibberish
]

for question in test_cases:
    print(f"\n{'─'*80}")
    print(f"Question: \"{question}\"")
    print(f"{'─'*80}")
    
    result = predictor.predict(question)
    
    print(f"Intent: {result['intent']}")
    print(f"Confidence: {result['confidence']:.1f}%")
    print(f"Response: {result['response'][:200]}...")
    
    # Show what answer should be from database
    if result['intent']:
        try:
            correct_training = TrainingData.objects.filter(
                intent__name=result['intent'],
                is_active=True
            ).first()
            if correct_training:
                print(f"\n✓ Expected answer from DB: {correct_training.answer[:100]}...")
            else:
                print(f"\n⚠️  No training data found for intent: {result['intent']}")
        except:
            pass

print("\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80)