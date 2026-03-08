import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'campusbot.settings')
django.setup()

from chatbot.models import Category, Intent, TrainingData

def populate_training_data():

    print("Populating academic training data (10 questions per intent)...")
    print("=" * 60)

    training_data = [
        {
    "category": "School Information",
    "intent": "official_website",
    "description": "Official website of Lipa City Colleges",
    "questions": [

        # English
        "What is the official website of Lipa City Colleges?",
        "Do you have an official website?",
        "Where can I visit the school website?",
        "What is the website of LCC?",
        "Can you give me the official website link?",

        # Filipino
        "Ano official website ng Lipa City Colleges?",
        "May website ba ang LCC?",
        "Saan ko makikita ang website ng school?",
        "Ano link ng website ng Lipa City Colleges?",
        "Paano ko mapupuntahan ang website ng LCC?"
    ],

    "answer": """
For more info, you can visit the official website of Lipa City Colleges here:

https://www.lipacitycolleges.edu.ph/
https://www.facebook.com/LeadershipCompetenceCommitment

"""
}

    ]

    total_intents = 0
    total_training = 0

    for data in training_data:

        # Create or get category
        category, _ = Category.objects.get_or_create(
            name=data["category"]
        )

        # Create or get intent
        intent, created = Intent.objects.get_or_create(
            name=data["intent"],
            defaults={
                "description": data["description"],
                "category": category
            }
        )

        if created:
            total_intents += 1
            print(f"✓ Created intent: {intent.name}")

        # Create training data
        training = TrainingData.objects.create(
            intent=intent,
            questions_data=data["questions"],
            answer=data["answer"],
            is_active=True
        )

        total_training += 1
        print(f"  + Added training data for intent: {intent.name}")

    print("\n" + "=" * 60)
    print("ACADEMIC TRAINING DATA POPULATED!")
    print(f"Total Intents: {total_intents}")
    print(f"Total Training Entries: {total_training}")
    print("=" * 60)


if __name__ == "__main__":
    populate_training_data()