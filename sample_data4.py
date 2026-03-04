# populate_training_data.py
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'campusbot.settings')
django.setup()

from chatbot.models import Intent, TrainingData


def populate_training_data():
    print("Populating training data (Rasa-style NLU)...")
    print("=" * 60)

    training_data = [

        # ===== GREETING =====
        {
            "intent": "greeting",
            "description": "User greetings",
            "answer": "Hello! Welcome to Campus Services. How can I help you today?",
            "questions": [
                "Hi",
                "Hello",
                "Hey",
                "Good morning",
                "Good afternoon",
                "Hi there",
                "Hello, I need help"
            ]
        },

        # ===== LIBRARY HOURS =====
        {
            "intent": "ask_library_hours",
            "description": "Ask about library operating hours",
            "answer": (
                "The library is open Monday to Friday from 8:00 AM to 10:00 PM, "
                "Saturday from 9:00 AM to 6:00 PM, and Sunday from 12:00 PM to 8:00 PM."
            ),
            "questions": [
                "What are the library hours?",
                "When does the library open?",
                "When does the library close?",
                "Is the library open on weekends?",
                "Library operating hours",
                "What time does the library open?",
                "What time does the library close?"
            ]
        },

        # ===== REGISTRATION =====
        {
            "intent": "ask_registration_process",
            "description": "Ask how to register for classes",
            "answer": (
                "You can register for classes through the Student Portal. "
                "Log in using your student account, go to the Registration section, "
                "select your courses, and submit your enrollment."
            ),
            "questions": [
                "How do I register for classes?",
                "How can I enroll in subjects?",
                "Course registration process",
                "How to register for subjects",
                "How do I enroll this semester?"
            ]
        },

        # ===== TUITION FEES =====
        {
            "intent": "ask_tuition_fees",
            "description": "Ask about tuition and fees",
            "answer": (
                "Tuition fees depend on your program and year level. "
                "You can view the official fee breakdown in the Student Portal "
                "or contact the Finance Office for detailed information."
            ),
            "questions": [
                "How much is the tuition?",
                "What are the tuition fees?",
                "How much do I need to pay?",
                "Tuition cost",
                "How much is the school fee?"
            ]
        },

        # ===== IT SUPPORT =====
        {
            "intent": "ask_it_support",
            "description": "Ask for IT or technical support",
            "answer": (
                "For IT support, you can visit the IT Office or contact the Help Desk. "
                "They can assist with Wi-Fi access, account issues, and system problems."
            ),
            "questions": [
                "I forgot my password",
                "My student account is not working",
                "WiFi is not connecting",
                "I need IT support",
                "Technical problem"
            ]
        },

        # ===== GOODBYE =====
        {
            "intent": "goodbye",
            "description": "Conversation ending",
            "answer": "You're welcome! If you have more questions, feel free to ask. Have a great day!",
            "questions": [
                "Thank you",
                "Thanks",
                "Goodbye",
                "Bye",
                "That's all",
                "No more questions"
            ]
        },
    ]

    total_intents = 0
    total_samples = 0

    for item in training_data:
        intent, created = Intent.objects.get_or_create(
            name=item["intent"],
            defaults={"description": item["description"]}
        )

        if created:
            total_intents += 1
            print(f"✓ Created intent: {intent.name}")
        else:
            print(f"→ Intent exists: {intent.name}")

        for question in item["questions"]:
            obj, created = TrainingData.objects.get_or_create(
                intent=intent,
                question=question,
                defaults={"answer": item["answer"]}
            )

            if created:
                total_samples += 1
                print(f"  + Added: {question}")

    print("=" * 60)
    print("DONE!")
    print(f"Total intents added: {total_intents}")
    print(f"Total training samples added: {total_samples}")
    print("=" * 60)


if __name__ == "__main__":
    populate_training_data()
