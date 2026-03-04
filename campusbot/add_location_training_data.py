import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'campusbot.settings')
django.setup()

from chatbot.models import Intent, TrainingData

def add_location_intent():
    """Add training data for room location queries"""
    
    # Create intent
    intent, created = Intent.objects.get_or_create(
        name='ask_room_location',
        defaults={'description': 'Questions about room/office locations'}
    )
    
    if created:
        print("✓ Created intent: ask_room_location")
    else:
        print("→ Intent already exists: ask_room_location")
    
    # Generic questions (room number will be extracted)
    questions = [
        "Where is room 401?",
        "How do I find room 402?",
        "Room 403 location?",
        "Where can I find room 202?",
        "Directions to room 401",
        "How to get to room 402?",
        "Where is the smart lab?",
        "Find the smart lab",
        "Smart lab location?",
        "Where is the library?",
        "Library location?",
        "Where can I find the registrar?",
        "Registrar office location?",
        "Where is the cafeteria?",
        "How do I get to the cafeteria?",
        "Find building A room 401",
        "Location of room 402",
        "Which building is room 403 in?",
        "What floor is room 401 on?",
        "Where is CS lab?",
        "Computer science lab location?",
        "Find physics lab",
        "Chemistry lab where?",
    ]
    
    # Generic answer (actual answer generated dynamically)
    answer = (
        "I can help you find campus locations! "
        "Please specify the room number or location name you're looking for. "
        "For example: 'Where is room 401?' or 'Find the Smart Lab'"
    )
    
    # Check if training data exists
    existing = TrainingData.objects.filter(intent=intent, is_active=True).first()
    
    if existing:
        existing.set_questions(questions)
        existing.answer = answer
        existing.save()
        print(f"✓ Updated training data with {len(questions)} questions")
    else:
        training = TrainingData.objects.create(
            intent=intent,
            answer=answer,
            is_active=True
        )
        training.set_questions(questions)
        print(f"✓ Created training data with {len(questions)} questions")
    
    print("\nNow retrain the model via admin dashboard!")

if __name__ == '__main__':
    add_location_intent()