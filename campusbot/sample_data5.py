import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'campusbot.settings')
django.setup()

from chatbot.models import Intent, TrainingData

def populate_training_data():
    """Populate with single-row format"""
    
    print("Starting to populate training data (single-row format)...")
    print("=" * 60)
    
    training_data = [
        {
            'intent': 'greeting',
            'description': 'User greetings and conversation starters',
            'questions': [
                'Hi',
                'Hello',
                'Hey',
                'Good morning',
                'Good afternoon',
                'Hey there',
                'Hi there',
                'Greetings',
            ],
            'answer': 'Hello! Welcome to Campus Services. How can I help you today?'
        },
        {
            'intent': 'library_hours',
            'description': 'Information about library operating hours',
            'questions': [
                'What are the library hours?',
                'When does the library open?',
                'Library operating hours',
                'Is the library open on weekends?',
                'When does the library close?',
                'What time does library open today?',
                'Library schedule',
                'Library hours?',
                'What time library close?',
                'Is library open today?',
            ],
            'answer': 'The library is open Monday-Friday 8:00 AM - 10:00 PM, Saturday 9:00 AM - 6:00 PM, and Sunday 12:00 PM - 8:00 PM. Hours may vary during holidays and exam periods.'
        },
        {
            'intent': 'registration',
            'description': 'Course registration and enrollment',
            'questions': [
                'How do I register for classes?',
                'When is registration?',
                'Course enrollment help',
                'Can I add a class?',
                'How to enroll in courses',
                'Register for classes',
                'Class registration process',
                'How to register?',
                'Registration dates',
                'When can I register?',
            ],
            'answer': 'You can register for classes through the Student Portal. Log in with your student credentials, go to "Registration" tab, search for courses, and click "Add to Cart". Make sure to check prerequisites and enrollment dates.'
        },
        {
            'intent': 'tuition_fees',
            'description': 'Tuition costs and payment information',
            'questions': [
                'How much is tuition?',
                'What are the fees?',
                'Payment options',
                'When is tuition due?',
                'Can I pay tuition in installments?',
                'Tuition cost',
                'How do I pay my tuition?',
                'Tuition payment',
                'What are the costs?',
                'Semester fees',
            ],
            'answer': 'Tuition varies by program. Undergraduate full-time is approximately $15,000/semester. Payment can be made online through the Student Portal, by mail, or in person at the Bursar\'s Office. We offer payment plans - contact bursar@campus.edu or (555) 123-4567.'
        },
        {
            'intent': 'financial_aid',
            'description': 'Financial aid and scholarships',
            'questions': [
                'How do I apply for financial aid?',
                'Are there scholarships available?',
                'FAFSA help',
                'Student loans information',
                'Scholarship deadlines',
                'Financial aid application',
                'How to get financial aid?',
                'Scholarship opportunities',
                'Financial assistance',
                'Aid application',
            ],
            'answer': 'Complete the FAFSA at fafsa.gov (school code: 123456). Priority deadline is March 1st. We offer merit and need-based scholarships - visit our Financial Aid website for available scholarships and requirements.'
        },
        {
            'intent': 'goodbye',
            'description': 'Conversation endings',
            'questions': [
                'Thanks',
                'Thank you',
                'Goodbye',
                'Bye',
                'See you',
                'That\'s all',
                'That\'s all I needed',
                'Thanks for your help',
            ],
            'answer': 'You\'re welcome! If you have more questions, feel free to ask. Have a great day!'
        },
    ]
    
    total_intents = 0
    total_questions = 0
    
    for data in training_data:
        # Create or get intent
        intent, created = Intent.objects.get_or_create(
            name=data['intent'],
            defaults={'description': data['description']}
        )
        
        if created:
            print(f"✓ Created intent: {intent.name}")
            total_intents += 1
        else:
            print(f"→ Intent exists: {intent.name}")
        
        # Check if training data exists
        existing = TrainingData.objects.filter(intent=intent, is_active=True).first()
        
        if existing:
            # Update existing
            existing.set_questions(data['questions'])
            existing.answer = data['answer']
            existing.save()
            print(f"  ✓ Updated with {len(data['questions'])} questions")
        else:
            # Create new
            training = TrainingData.objects.create(
                intent=intent,
                answer=data['answer'],
                is_active=True
            )
            training.set_questions(data['questions'])
            print(f"  ✓ Created with {len(data['questions'])} questions")
        
        total_questions += len(data['questions'])
    
    print("\n" + "=" * 60)
    print("TRAINING DATA POPULATION COMPLETE!")
    print("=" * 60)
    print(f"Intents: {total_intents} created")
    print(f"Total Questions: {total_questions}")
    print(f"Total Rows: {TrainingData.objects.filter(is_active=True).count()}")
    print("\nNow train your model via admin dashboard!")
    print("=" * 60)

if __name__ == '__main__':
    populate_training_data()