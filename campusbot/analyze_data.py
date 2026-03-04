# analyze_training_data.py
import os
import django
from collections import Counter

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'campusbot.settings')
django.setup()

from chatbot.models import TrainingData, Intent

def analyze_training_data():
    data = TrainingData.objects.filter(is_active=True).select_related('intent')
    
    if not data.exists():
        print("No training data found.")
        return
    
    # Count samples per intent
    intent_counter = Counter()
    intent_questions = {}
    
    for item in data:
        intent_name = item.intent.name
        intent_counter[intent_name] += 1
        if intent_name not in intent_questions:
            intent_questions[intent_name] = []
        intent_questions[intent_name].append(item.question)
    
    print("\n=== INTENT SAMPLE COUNT ===")
    for intent, count in intent_counter.most_common():
        status = "⚠️ Too few samples!" if count < 5 else "✓ Enough samples"
        print(f"{intent}: {count} samples {status}")
    
    print("\n=== POTENTIAL PROBLEMATIC QUESTIONS ===")
    for intent, questions in intent_questions.items():
        if len(questions) < 5:
            print(f"\nIntent: {intent} (only {len(questions)} samples)")
            for q in questions:
                print(f"  - {q}")
        else:
            # Check for very short questions
            short_questions = [q for q in questions if len(q.split()) <= 2]
            if short_questions:
                print(f"\nIntent: {intent} (contains very short questions)")
                for q in short_questions:
                    print(f"  - {q}")

    print("\n=== SUGGESTIONS ===")
    print("- Add at least 5–10 training samples per intent.")
    print("- Paraphrase questions with different words but same meaning.")
    print("- Combine intents if they are very similar with few samples.")
    print("- Avoid too many single-word questions for Naive Bayes.")

if __name__ == '__main__':
    analyze_training_data()
