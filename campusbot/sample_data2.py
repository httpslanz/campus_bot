# populate_training_data_v2.py
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'campusbot.settings')
django.setup()

from chatbot.models import Intent, TrainingData, IntentResponse

def populate_training_data_clean():
    """Populate with separated questions and answers"""
    
    print("Starting to populate training data (clean version)...")
    print("=" * 60)
    
    training_data = [
        {
            'intent': 'library_hours',
            'description': 'Information about library operating hours',
            'answer': 'The library is open Monday-Friday 8:00 AM - 10:00 PM, Saturday 9:00 AM - 6:00 PM, and Sunday 12:00 PM - 8:00 PM. Hours may vary during holidays and exam periods.',
            'questions': [
                'What are the library hours?',
                'When does the library open?',
                'Library operating hours',
                'Is the library open on weekends?',
                'When does the library close?',
                'What time does library open today?',
                'Library schedule',
            ]
        },
        {
            'intent': 'registration',
            'description': 'Course registration and enrollment information',
            'answer': 'You can register for classes through the Student Portal. Log in with your student credentials, go to "Registration" tab, search for courses, and click "Add to Cart". Make sure to check prerequisites and enrollment dates.',
            'questions': [
                'How do I register for classes?',
                'When is registration?',
                'Course enrollment help',
                'Can I add a class?',
                'How to enroll in courses',
                'Register for classes',
                'Class registration process',
            ]
        },
        {
            'intent': 'tuition_fees',
            'description': 'Tuition costs and payment information',
            'answer': 'Tuition varies by program. Undergraduate full-time is approximately $15,000/semester. Payment can be made online through the Student Portal, by mail, or in person at the Bursar\'s Office. We offer payment plans - contact bursar@campus.edu or (555) 123-4567.',
            'questions': [
                'How much is tuition?',
                'What are the fees?',
                'Payment options',
                'When is tuition due?',
                'Can I pay tuition in installments?',
                'Tuition cost',
                'How do I pay my tuition?',
            ]
        },
        {
            'intent': 'financial_aid',
            'description': 'Financial aid and scholarship information',
            'answer': 'Complete the FAFSA at fafsa.gov (school code: 123456). Priority deadline is March 1st. We offer merit and need-based scholarships - visit our Financial Aid website for available scholarships and requirements.',
            'questions': [
                'How do I apply for financial aid?',
                'Are there scholarships available?',
                'FAFSA help',
                'Student loans information',
                'Scholarship deadlines',
                'Financial aid application',
                'How to get financial aid?',
            ]
        },
        {
            'intent': 'campus_map',
            'description': 'Campus navigation and building locations',
            'answer': 'View our interactive campus map at campus.edu/map or pick up a printed map at the Information Desk in the Student Center. The map shows all buildings, parking lots, and campus amenities. You can also download our mobile campus app with GPS navigation.',
            'questions': [
                'Where is the student center?',
                'Campus map',
                'How do I find a building?',
                'Where is the library?',
                'Parking locations',
                'Campus building locations',
                'How do I navigate campus?',
            ]
        },
        {
            'intent': 'dining_services',
            'description': 'Campus dining halls and meal plan information',
            'answer': 'Main Dining Hall: Mon-Fri 7AM-8PM, weekends 9AM-7PM. Student Center Cafe: Mon-Fri 7AM-10PM. Meal plans include Unlimited, 14 meals/week, 10 meals/week. All options have vegetarian, vegan, and gluten-free choices. Visit dining.campus.edu for menus.',
            'questions': [
                'What are the dining hall hours?',
                'Meal plan options',
                'Where can I eat on campus?',
                'Do you have vegetarian options?',
                'How do I add dining dollars?',
                'Campus food options',
                'Dining hall menu',
            ]
        },
        {
            'intent': 'it_support',
            'description': 'Technology help and IT services',
            'answer': 'Reset password at password.campus.edu or contact IT Help Desk at (555) 123-HELP. Connect to "Campus-WiFi" with student credentials. Student email: username@student.campus.edu. Free software downloads at software.campus.edu. Print stations in Library and Student Center.',
            'questions': [
                'I forgot my password',
                'How do I connect to WiFi?',
                'Email not working',
                'Software downloads',
                'Printing on campus',
                'IT help',
                'Tech support',
                'Password reset',
            ]
        },
        {
            'intent': 'greeting',
            'description': 'User greetings and conversation starters',
            'answer': 'Hello! Welcome to Campus Services. How can I help you today?',
            'questions': [
                'Hi',
                'Hello',
                'Hey',
                'Good morning',
                'Good afternoon',
                'Hey there',
                'Hi there',
                'Greetings',
            ]
        },
        {
            'intent': 'goodbye',
            'description': 'Conversation endings',
            'answer': 'You\'re welcome! Have a great day. Feel free to return if you have more questions!',
            'questions': [
                'Thanks',
                'Thank you',
                'Goodbye',
                'Bye',
                'See you',
                'That\'s all',
                'That\'s all I needed',
            ]
        },
    ]
    
    total_intents = 0
    total_questions = 0
    total_answers = 0
    
    for data in training_data:
        # Create intent
        intent, created = Intent.objects.get_or_create(
            name=data['intent'],
            defaults={'description': data['description']}
        )
        
        if created:
            print(f"✓ Created intent: {intent.name}")
            total_intents += 1
        else:
            print(f"→ Intent exists: {intent.name}")
        
        # Create answer (only once per intent)
        answer, created = IntentResponse.objects.get_or_create(
            intent=intent,
            answer=data['answer'],
            defaults={'is_default': True, 'priority': 1}
        )
        
        if created:
            total_answers += 1
            print(f"  ✓ Added answer")
        
        # Create questions (for training)
        for question in data['questions']:
            training, created = TrainingData.objects.get_or_create(
                intent=intent,
                question=question,
                defaults={'answer': data['answer']}  # Keep for backward compatibility
            )
            
            if created:
                total_questions += 1
                print(f"  + {question}")
    
    print("\n" + "=" * 60)
    print("CLEAN TRAINING DATA POPULATION COMPLETE!")
    print("=" * 60)
    print(f"Intents Created: {total_intents}")
    print(f"Questions Added: {total_questions}")
    print(f"Answers Stored: {total_answers}")
    print("\nStructure:")
    print("  - Each intent has ONE answer in IntentResponse table")
    print("  - Each intent has MULTIPLE questions in TrainingData table")
    print("  - Model learns from questions")
    print("  - System fetches answer dynamically")
    print("=" * 60)

if __name__ == '__main__':
    populate_training_data_clean()