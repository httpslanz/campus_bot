import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "campusbot.settings")
django.setup()

from chatbot.models import Intent, TrainingData


def populate_training_data():
    print("Populating training data...")
    print("=" * 60)

    training_data = {
        "greeting": [
            "Hello",
            "Hi",
            "Good morning",
            "Good afternoon",
            "Good evening"
        ],
        "goodbye": [
            "Goodbye",
            "Bye",
            "See you later",
            "Thanks, bye",
            "Talk to you later"
        ],
        "ask_scholarship": [
            "Where can I apply for scholarship?",
            "How do I apply for scholarship?",
            "Are there scholarships available?",
            "What scholarships are offered?",
            "How can I get financial aid?"
        ],
        "ask_when_is_enrollment": [
            "When is enrollment?",
            "When does enrollment start?",
            "What is the enrollment schedule?",
            "Is enrollment open now?",
            "Until when is enrollment?"
        ],
        "tuition_fees": [
            "How much is the tuition fee?",
            "What is the tuition fee?",
            "How much do I need to pay?",
            "What are the school fees?",
            "How much is the total payment?"
        ],
        
        "ask_founder": [
            "Who is the founder of Lipa City Colleges?",
            "Who founded Lipa City Colleges?",
            "Who started Lipa City Colleges?",
            "Who established Lipa City Colleges?",
            "Who created Lipa City Colleges?"
        ],

        "registration": [
            "How do I enroll?",
            "How can I register my subjects?",
            "What is the enrollment process?",
            "How do I register for classes?",
            "Where can I enroll for this semester?"
        ],

        "library_hours": [
            "What are the library hours?",
            "What time does the library open?",
            "What time does the library close?",
            "Is the library open today?",
            "When is the library available?"
        ],
    }

    created_count = 0

    for intent_name, questions in training_data.items():

        try:
            intent = Intent.objects.get(name=intent_name)
        except Intent.DoesNotExist:
            print(f"⚠ Intent not found: {intent_name}")
            continue

        # Create ONE TrainingData record per intent
        obj, created = TrainingData.objects.update_or_create(
            intent=intent,
            defaults={
                "questions_data": questions,  # ✅ FIXED FIELD
                "answer": "Sample answer for now.",
                "is_active": True,
                "is_reviewed": True
            }
        )

        if created:
            print(f"✓ Created training data for {intent.name}")
            created_count += 1
        else:
            print(f"→ Updated training data for {intent.name}")

    print("\n" + "=" * 60)
    print(f"✓ Processed {len(training_data)} intents")
    print(f"✓ Total training records: {TrainingData.objects.count()}")
    print("=" * 60)


if __name__ == "__main__":
    populate_training_data()