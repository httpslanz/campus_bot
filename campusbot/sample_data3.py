# populate_training_data.py
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'campusbot.settings')
django.setup()

from chatbot.models import Intent, TrainingData


def populate_training_data():
    """Populate database with sample training data for campus chatbot"""

    print("Starting to populate training data...")
    print("=" * 60)

    training_data = [

        # ===== GREETING =====
        {
            'intent': 'greeting',
            'description': 'User greetings and conversation starters',
            'samples': [
                {'question': 'Hi', 'answer': 'Hello! How can I help you with campus services today?'},
                {'question': 'Hello', 'answer': 'Hello! How can I help you with campus services today?'},
                {'question': 'Hey', 'answer': 'Hello! How can I help you with campus services today?'},
                {'question': 'Good morning', 'answer': 'Good morning! How can I assist you today?'},
                {'question': 'Good afternoon', 'answer': 'Good afternoon! How can I assist you today?'},
            ]
        },

        # ===== ENROLLMENT =====
        {
            'intent': 'enrollment',
            'description': 'Enrollment and registration inquiries',
            'samples': [
                {'question': 'How can I enroll?', 'answer': 'You can enroll through the registrar’s office or the online enrollment system.'},
                {'question': 'What is the enrollment process?', 'answer': 'You can enroll through the registrar’s office or the online enrollment system.'},
                {'question': 'Where do I enroll?', 'answer': 'Enrollment is handled by the registrar’s office or via the student portal.'},
                {'question': 'When is enrollment?', 'answer': 'Enrollment schedules are announced by the registrar each semester.'},
                {'question': 'How do I register for classes?', 'answer': 'You may register for classes using the student portal during the enrollment period.'},
            ]
        },

        # ===== TUITION AND FEES =====
        {
            'intent': 'tuition_fees',
            'description': 'Tuition and school fees information',
            'samples': [
                {'question': 'How much is the tuition?', 'answer': 'Tuition fees depend on your program and year level. Please contact the cashier or registrar for details.'},
                {'question': 'What are the school fees?', 'answer': 'School fees include tuition, miscellaneous, and other charges. Visit the cashier for a breakdown.'},
                {'question': 'Where can I pay my tuition?', 'answer': 'Tuition payments can be made at the cashier or through approved online payment methods.'},
                {'question': 'Is there a payment deadline?', 'answer': 'Yes, payment deadlines are set each semester. Please check official announcements.'},
                {'question': 'Can I pay in installments?', 'answer': 'Installment plans may be available. Please inquire at the cashier’s office.'},
            ]
        },

        # ===== OFFICE HOURS =====
        {
            'intent': 'office_hours',
            'description': 'Campus office operating hours',
            'samples': [
                {'question': 'What are the office hours?', 'answer': 'Campus offices are open from 8:00 AM to 5:00 PM, Monday to Friday.'},
                {'question': 'Is the registrar open today?', 'answer': 'The registrar is open from 8:00 AM to 5:00 PM on weekdays.'},
                {'question': 'What time does the office close?', 'answer': 'Most offices close at 5:00 PM.'},
                {'question': 'Are offices open on weekends?', 'answer': 'Campus offices are usually closed on weekends.'},
                {'question': 'Office schedule', 'answer': 'Offices operate from 8:00 AM to 5:00 PM, Monday to Friday.'},
            ]
        },

        # ===== DOCUMENT REQUESTS =====
        {
            'intent': 'request_documents',
            'description': 'Requests for official school documents',
            'samples': [
                {'question': 'How do I request TOR?', 'answer': 'You may request your Transcript of Records at the registrar’s office.'},
                {'question': 'I need a certificate of enrollment', 'answer': 'Certificates can be requested from the registrar’s office.'},
                {'question': 'How can I get my TOR?', 'answer': 'Submit a request for your Transcript of Records at the registrar.'},
                {'question': 'Request certificate', 'answer': 'You may request certificates at the registrar’s office.'},
                {'question': 'Where do I get school documents?', 'answer': 'All official documents are issued by the registrar’s office.'},
            ]
        },

        # ===== CAMPUS LOCATION =====
        {
            'intent': 'campus_location',
            'description': 'Campus and office locations',
            'samples': [
                {'question': 'Where is the registrar office?', 'answer': 'The registrar’s office is located in the administration building.'},
                {'question': 'Where is the admin office?', 'answer': 'The administration office is in the main campus building.'},
                {'question': 'Location of registrar', 'answer': 'The registrar is located in the administration building.'},
                {'question': 'Where can I find the cashier?', 'answer': 'The cashier’s office is located near the registrar.'},
                {'question': 'Campus map', 'answer': 'You may view the campus map at the administration office or official website.'},
            ]
        },

        # ===== GOODBYE =====
        {
            'intent': 'goodbye',
            'description': 'Conversation endings',
            'samples': [
                {'question': 'Thank you', 'answer': 'You’re welcome! Let me know if you need anything else.'},
                {'question': 'Thanks', 'answer': 'You’re welcome! Have a great day.'},
                {'question': 'Goodbye', 'answer': 'Goodbye! Feel free to come back if you have more questions.'},
                {'question': 'Bye', 'answer': 'Goodbye! Take care.'},
                {'question': 'That is all', 'answer': 'Alright! I’m glad I could help.'},
            ]
        },

        # ===== FALLBACK =====
        {
            'intent': 'fallback',
            'description': 'Unknown or unsupported queries',
            'samples': [
                {'question': 'asdfgh', 'answer': 'I’m not sure I understood that. You may ask about enrollment, fees, or office hours.'},
                {'question': 'Tell me a joke', 'answer': 'I’m here to help with campus services. Try asking about enrollment or documents.'},
                {'question': 'What is your favorite food?', 'answer': 'I’m here to assist with campus-related questions.'},
                {'question': 'Random question', 'answer': 'I may not have information on that. Please ask about campus services.'},
                {'question': 'Who are you?', 'answer': 'I’m the campus services chatbot, here to help you.'},
            ]
        },
    ]

    total_intents = 0
    total_samples = 0

    for intent_data in training_data:
        intent, created = Intent.objects.get_or_create(
            name=intent_data['intent'],
            defaults={'description': intent_data['description']}
        )

        if created:
            total_intents += 1
            print(f"✓ Created intent: {intent.name}")
        else:
            print(f"→ Intent already exists: {intent.name}")

        for sample in intent_data['samples']:
            obj, created = TrainingData.objects.get_or_create(
                intent=intent,
                question=sample['question'],
                defaults={'answer': sample['answer']}
            )

            if created:
                total_samples += 1
                print(f"  + Added: {sample['question']}")

    print("\n" + "=" * 60)
    print("TRAINING DATA POPULATION COMPLETE")
    print("=" * 60)
    print(f"Total intents created: {total_intents}")
    print(f"Total samples added: {total_samples}")
    print("=" * 60)


if __name__ == "__main__":
    populate_training_data()
