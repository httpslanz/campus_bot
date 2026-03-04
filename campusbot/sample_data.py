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
    
    # Define training data structure
    training_data = [
        # ===== GREETING =====
        {
            'intent': 'greeting',
            'description': 'User greetings and conversation starters',
            'samples': [
                {
                    'question': 'Hi',
                    'answer': 'Hello! Welcome to Campus Services. How can I help you today?'
                },
                {
                    'question': 'Hello',
                    'answer': 'Hello! Welcome to Campus Services. How can I help you today?'
                },
                {
                    'question': 'Hey there',
                    'answer': 'Hello! Welcome to Campus Services. How can I help you today?'
                },
                {
                    'question': 'Good morning',
                    'answer': 'Good morning! How can I assist you with campus services today?'
                },
                {
                    'question': 'Hi, I need help',
                    'answer': 'Hello! I\'m here to help. What do you need assistance with?'
                },
            ]
        },
        
        # ===== LIBRARY HOURS =====
        {
            'intent': 'library_hours',
            'description': 'Information about library operating hours',
            'samples': [
                {
                    'question': 'What are the library hours?',
                    'answer': 'The library is open Monday-Friday 8:00 AM - 10:00 PM, Saturday 9:00 AM - 6:00 PM, and Sunday 12:00 PM - 8:00 PM. Hours may vary during holidays and exam periods.'
                },
                {
                    'question': 'When does the library open?',
                    'answer': 'The library is open Monday-Friday 8:00 AM - 10:00 PM, Saturday 9:00 AM - 6:00 PM, and Sunday 12:00 PM - 8:00 PM. Hours may vary during holidays and exam periods.'
                },
                {
                    'question': 'Library operating hours',
                    'answer': 'The library is open Monday-Friday 8:00 AM - 10:00 PM, Saturday 9:00 AM - 6:00 PM, and Sunday 12:00 PM - 8:00 PM. Hours may vary during holidays and exam periods.'
                },
                {
                    'question': 'Is the library open on weekends?',
                    'answer': 'Yes! The library is open Saturday 9:00 AM - 6:00 PM and Sunday 12:00 PM - 8:00 PM.'
                },
                {
                    'question': 'When does the library close?',
                    'answer': 'The library closes at 10:00 PM on weekdays, 6:00 PM on Saturday, and 8:00 PM on Sunday.'
                },
            ]
        },
        
        # ===== REGISTRATION =====
        {
            'intent': 'registration',
            'description': 'Course registration and enrollment information',
            'samples': [
                {
                    'question': 'How do I register for classes?',
                    'answer': 'You can register for classes through the Student Portal. Log in with your student credentials, go to "Registration" tab, search for courses, and click "Add to Cart". Make sure to check prerequisites and enrollment dates.'
                },
                {
                    'question': 'When is registration?',
                    'answer': 'Registration dates vary by class level. Check your student portal for your specific registration window. Generally, seniors register first, followed by juniors, sophomores, and freshmen.'
                },
                {
                    'question': 'Course enrollment help',
                    'answer': 'You can register for classes through the Student Portal. Log in with your student credentials, go to "Registration" tab, search for courses, and click "Add to Cart". Make sure to check prerequisites and enrollment dates.'
                },
                {
                    'question': 'Can I add a class?',
                    'answer': 'Yes, you can add classes during the registration period or add/drop period. Log into the Student Portal and navigate to the Registration section. Contact the Registrar\'s Office if you need an override.'
                },
                {
                    'question': 'How to enroll in courses',
                    'answer': 'You can register for classes through the Student Portal. Log in with your student credentials, go to "Registration" tab, search for courses, and click "Add to Cart". Make sure to check prerequisites and enrollment dates.'
                },
            ]
        },
        
        # ===== TUITION AND FEES =====
        {
            'intent': 'tuition_fees',
            'description': 'Tuition costs and payment information',
            'samples': [
                {
                    'question': 'How much is tuition?',
                    'answer': 'Tuition varies by program and enrollment status. For undergraduate students, full-time tuition is approximately $15,000 per semester. Graduate programs range from $18,000-$25,000 per semester. Visit the Bursar\'s Office website for detailed fee schedules.'
                },
                {
                    'question': 'What are the fees?',
                    'answer': 'In addition to tuition, students pay various fees including: student activity fee ($200/semester), technology fee ($150/semester), and health services fee ($100/semester). Additional course-specific fees may apply.'
                },
                {
                    'question': 'Payment options',
                    'answer': 'Payment can be made online through the Student Portal, by mail, or in person at the Bursar\'s Office. We offer payment plans - contact the Bursar\'s Office at bursar@campus.edu or call (555) 123-4567.'
                },
                {
                    'question': 'When is tuition due?',
                    'answer': 'Tuition is due two weeks before the start of each semester. Fall semester payment is typically due in early August, and Spring semester payment is due in early January. Check your student account for specific due dates.'
                },
                {
                    'question': 'Can I pay tuition in installments?',
                    'answer': 'Yes! We offer a payment plan option. Contact the Bursar\'s Office at bursar@campus.edu or call (555) 123-4567 to set up a payment plan.'
                },
            ]
        },
        
        # ===== FINANCIAL AID =====
        {
            'intent': 'financial_aid',
            'description': 'Financial aid and scholarship information',
            'samples': [
                {
                    'question': 'How do I apply for financial aid?',
                    'answer': 'To apply for financial aid, complete the FAFSA (Free Application for Federal Student Aid) at fafsa.gov. Our school code is 123456. The priority deadline is March 1st for the following academic year. Visit the Financial Aid Office for additional assistance.'
                },
                {
                    'question': 'Are there scholarships available?',
                    'answer': 'Yes! We offer merit-based and need-based scholarships. Visit our Financial Aid website to browse available scholarships and application requirements. Many scholarships have deadlines in February for the following academic year.'
                },
                {
                    'question': 'FAFSA help',
                    'answer': 'To apply for financial aid, complete the FAFSA (Free Application for Federal Student Aid) at fafsa.gov. Our school code is 123456. The Financial Aid Office offers FAFSA completion workshops. Check our website for upcoming dates or schedule an appointment.'
                },
                {
                    'question': 'Student loans information',
                    'answer': 'Federal student loans are available through the FAFSA. After completing your FAFSA, you\'ll receive a financial aid award letter detailing your loan eligibility. Contact the Financial Aid Office at finaid@campus.edu for more information.'
                },
                {
                    'question': 'Scholarship deadlines',
                    'answer': 'Scholarship deadlines vary by scholarship. Most institutional scholarships have deadlines in February. Check the Financial Aid website for a complete list of scholarships and their specific deadlines.'
                },
            ]
        },
        
        # ===== CAMPUS MAP =====
        {
            'intent': 'campus_map',
            'description': 'Campus navigation and building locations',
            'samples': [
                {
                    'question': 'Where is the student center?',
                    'answer': 'The Student Center is located in the heart of campus at 100 Campus Drive. It\'s the large brick building near the main fountain. You can view the interactive campus map at campus.edu/map'
                },
                {
                    'question': 'Campus map',
                    'answer': 'You can view our interactive campus map at campus.edu/map or pick up a printed map at the Information Desk in the Student Center. The map shows all buildings, parking lots, and campus amenities.'
                },
                {
                    'question': 'How do I find a building?',
                    'answer': 'You can use our interactive campus map at campus.edu/map to locate any building. You can also download our mobile campus app which includes GPS navigation and building information.'
                },
                {
                    'question': 'Where is the library?',
                    'answer': 'The Main Library is located at 200 Scholar Way, across from the Student Center. It\'s the modern glass building with the distinctive dome. View the campus map at campus.edu/map for directions.'
                },
                {
                    'question': 'Parking locations',
                    'answer': 'Student parking is available in Lots A, B, and C. Lot A is near the dorms, Lot B is by the athletic center, and Lot C is near the academic buildings. View all parking options on the campus map at campus.edu/map'
                },
            ]
        },
        
        # ===== DINING SERVICES =====
        {
            'intent': 'dining_services',
            'description': 'Campus dining halls and meal plan information',
            'samples': [
                {
                    'question': 'What are the dining hall hours?',
                    'answer': 'The Main Dining Hall is open Monday-Friday 7:00 AM - 8:00 PM, and weekends 9:00 AM - 7:00 PM. The Student Center Cafe is open Monday-Friday 7:00 AM - 10:00 PM. Hours may vary during breaks.'
                },
                {
                    'question': 'Meal plan options',
                    'answer': 'We offer several meal plan options: Unlimited (all meals), 14 meals/week, 10 meals/week, and commuter plans. All plans include dining dollars. Visit dining.campus.edu or the Dining Services Office for details and pricing.'
                },
                {
                    'question': 'Where can I eat on campus?',
                    'answer': 'Campus dining options include: Main Dining Hall (all-you-care-to-eat), Student Center Cafe (grab-and-go), Library Cafe (coffee and snacks), and Food Court (various vendors). View all locations at dining.campus.edu'
                },
                {
                    'question': 'Do you have vegetarian options?',
                    'answer': 'Yes! All our dining locations offer vegetarian, vegan, and gluten-free options. The Main Dining Hall has dedicated stations for dietary restrictions. Visit dining.campus.edu for current menus and allergen information.'
                },
                {
                    'question': 'How do I add dining dollars?',
                    'answer': 'You can add dining dollars through the Student Portal under "Meal Plans" or visit the Dining Services Office in the Student Center. Dining dollars can be used at all campus dining locations.'
                },
            ]
        },
        
        # ===== HOUSING =====
        {
            'intent': 'housing',
            'description': 'Student housing and residence hall information',
            'samples': [
                {
                    'question': 'How do I apply for housing?',
                    'answer': 'Housing applications are available through the Student Portal starting in February. Log in, go to "Housing" section, and complete the application. Priority is given to first-year students. Housing deposit of $200 is required.'
                },
                {
                    'question': 'Residence hall information',
                    'answer': 'We have 6 residence halls: North Hall (freshmen), South Hall (freshmen), East Tower (upperclassmen), West Tower (upperclassmen), University Suites (juniors/seniors), and Graduate Housing. Visit housing.campus.edu for virtual tours.'
                },
                {
                    'question': 'When is housing selection?',
                    'answer': 'Housing selection occurs in April for the following academic year. You\'ll receive an email with your selection time based on your class year and housing lottery number. Check housing.campus.edu for the exact schedule.'
                },
                {
                    'question': 'Can I choose my roommate?',
                    'answer': 'Yes! You can request a specific roommate during the housing application process. Both students must request each other. You can also use our roommate matching system if you don\'t know anyone yet.'
                },
                {
                    'question': 'Housing costs',
                    'answer': 'Housing costs vary by residence hall and room type. Rates range from $3,500-$6,000 per semester. Traditional double rooms are the most affordable, while single rooms and suites cost more. Visit housing.campus.edu for detailed pricing.'
                },
            ]
        },
        
        # ===== IT SUPPORT =====
        {
            'intent': 'it_support',
            'description': 'Technology help and IT services',
            'samples': [
                {
                    'question': 'I forgot my password',
                    'answer': 'You can reset your password at password.campus.edu. You\'ll need your student ID and alternate email. If you have trouble, contact the IT Help Desk at (555) 123-HELP or visit the Tech Center in the Library.'
                },
                {
                    'question': 'How do I connect to WiFi?',
                    'answer': 'Connect to the "Campus-WiFi" network using your student credentials (username and password). For setup help, visit it.campus.edu/wifi or contact the IT Help Desk at (555) 123-HELP.'
                },
                {
                    'question': 'Email not working',
                    'answer': 'Your student email is username@student.campus.edu. Access it at email.campus.edu or set it up on your device. If you\'re having issues, contact IT Help Desk at (555) 123-HELP or email support@campus.edu'
                },
                {
                    'question': 'Software downloads',
                    'answer': 'Students can download free software including Microsoft Office, Adobe Creative Suite, and antivirus programs at software.campus.edu. Log in with your student credentials to access available downloads.'
                },
                {
                    'question': 'Printing on campus',
                    'answer': 'Print stations are located in the Library, Student Center, and all computer labs. Your student ID card includes $20 printing credit per semester. Additional credit can be added at print.campus.edu or campus kiosks.'
                },
            ]
        },
        
        # ===== HEALTH SERVICES =====
        {
            'intent': 'health_services',
            'description': 'Student health center and medical services',
            'samples': [
                {
                    'question': 'Health center hours',
                    'answer': 'The Student Health Center is open Monday-Friday 8:00 AM - 5:00 PM during the academic year. For after-hours emergencies, call Campus Security at (555) 123-9999 or dial 911.'
                },
                {
                    'question': 'How do I make a medical appointment?',
                    'answer': 'Schedule appointments online at health.campus.edu or call (555) 123-CARE. Walk-ins are accepted for urgent issues. The Health Center is located in the Wellness Building, Room 101.'
                },
                {
                    'question': 'Do I need health insurance?',
                    'answer': 'All students must have health insurance. You can use your own insurance or enroll in the Student Health Insurance Plan. Visit health.campus.edu/insurance for details and waiver information.'
                },
                {
                    'question': 'Mental health services',
                    'answer': 'Free counseling services are available to all students. Schedule an appointment at counseling.campus.edu or call (555) 123-TALK. Crisis support is available 24/7 through our partnership with the National Crisis Line.'
                },
                {
                    'question': 'Pharmacy on campus',
                    'answer': 'The campus pharmacy is located in the Health Center and fills most prescriptions. Hours are Monday-Friday 9:00 AM - 4:00 PM. Bring your prescription and insurance card.'
                },
            ]
        },
        
        # ===== BOOKSTORE =====
        {
            'intent': 'bookstore',
            'description': 'Campus bookstore information and textbooks',
            'samples': [
                {
                    'question': 'Bookstore hours',
                    'answer': 'The Campus Bookstore is open Monday-Friday 8:00 AM - 6:00 PM, and Saturday 10:00 AM - 4:00 PM. Extended hours during the first week of each semester. Visit bookstore.campus.edu for holiday hours.'
                },
                {
                    'question': 'How do I get my textbooks?',
                    'answer': 'Textbooks can be purchased at the Campus Bookstore or online at bookstore.campus.edu. We offer new, used, rental, and digital options. You can also check the course syllabus for ISBN numbers to shop elsewhere.'
                },
                {
                    'question': 'Can I rent textbooks?',
                    'answer': 'Yes! Textbook rentals are available for most courses at a significant discount. Rental books must be returned at the end of the semester. Visit bookstore.campus.edu or the store to browse rental options.'
                },
                {
                    'question': 'Textbook buyback',
                    'answer': 'The bookstore buys back textbooks during finals week each semester. Bring your books in good condition to the buyback counter. Prices depend on whether the book will be used next semester.'
                },
                {
                    'question': 'What else does the bookstore sell?',
                    'answer': 'In addition to textbooks, the bookstore sells school supplies, campus apparel, gifts, snacks, electronics, and course materials. We also have a print center for copying and binding services.'
                },
            ]
        },
        
        # ===== ATHLETICS =====
        {
            'intent': 'athletics',
            'description': 'Campus recreation and athletic facilities',
            'samples': [
                {
                    'question': 'Gym hours',
                    'answer': 'The Fitness Center is open Monday-Friday 6:00 AM - 11:00 PM, Saturday-Sunday 8:00 AM - 8:00 PM. Access is free for all students with your student ID. Visit recreation.campus.edu for group fitness schedules.'
                },
                {
                    'question': 'How do I join intramural sports?',
                    'answer': 'Intramural sports registration opens at the beginning of each semester. Visit recreation.campus.edu to see available sports and register your team. Popular sports include basketball, soccer, volleyball, and flag football.'
                },
                {
                    'question': 'Swimming pool hours',
                    'answer': 'The aquatic center is open for lap swimming Monday-Friday 6:00 AM - 8:00 AM and 5:00 PM - 9:00 PM, weekends 10:00 AM - 6:00 PM. Recreational swimming hours vary. Check recreation.campus.edu for the current schedule.'
                },
                {
                    'question': 'Where are the athletic games?',
                    'answer': 'Most games are held at Campus Stadium (football, soccer) and the Arena (basketball, volleyball). Check athletics.campus.edu for the full schedule. Student tickets are free with your student ID!'
                },
                {
                    'question': 'Personal training available?',
                    'answer': 'Yes! The Fitness Center offers personal training sessions. Rates are $30/session for students, or packages of 5 sessions for $125. Schedule at recreation.campus.edu or at the Fitness Center front desk.'
                },
            ]
        },
        
        # ===== CAREER SERVICES =====
        {
            'intent': 'career_services',
            'description': 'Career counseling and job search assistance',
            'samples': [
                {
                    'question': 'Resume help',
                    'answer': 'Career Services offers free resume reviews and workshops. Schedule an appointment at careers.campus.edu or visit our office in the Student Center, Room 300. Walk-in hours are Monday-Friday 1:00 PM - 4:00 PM.'
                },
                {
                    'question': 'Internship opportunities',
                    'answer': 'Browse internship postings on our career portal at jobs.campus.edu. Career Services also hosts internship fairs each semester. Check careers.campus.edu for upcoming events and application deadlines.'
                },
                {
                    'question': 'Job fair information',
                    'answer': 'We host career fairs each fall and spring semester. The Fall Career Fair is typically in October, and Spring in March. Register at careers.campus.edu to see attending employers and schedule interviews.'
                },
                {
                    'question': 'Interview preparation',
                    'answer': 'Career Services offers mock interviews and interview preparation workshops. Schedule an appointment at careers.campus.edu. We also have online resources including interview questions by industry and video tutorials.'
                },
                {
                    'question': 'How do I find a job after graduation?',
                    'answer': 'Career Services helps with job searches even after graduation! Use our job portal at jobs.campus.edu, attend networking events, and schedule advising appointments. We also offer alumni career services.'
                },
            ]
        },
        
        # ===== GOODBYE =====
        {
            'intent': 'goodbye',
            'description': 'Conversation endings and farewells',
            'samples': [
                {
                    'question': 'Thanks',
                    'answer': 'You\'re welcome! If you need anything else, feel free to ask. Have a great day!'
                },
                {
                    'question': 'Thank you',
                    'answer': 'You\'re welcome! If you need anything else, feel free to ask. Have a great day!'
                },
                {
                    'question': 'Goodbye',
                    'answer': 'Goodbye! Have a wonderful day. Feel free to return if you have more questions!'
                },
                {
                    'question': 'Bye',
                    'answer': 'Goodbye! Have a wonderful day. Feel free to return if you have more questions!'
                },
                {
                    'question': 'That\'s all I needed',
                    'answer': 'Great! I\'m glad I could help. Have a great day, and don\'t hesitate to reach out if you need anything else!'
                },
            ]
        },
    ]
    
    # Populate the database
    total_intents = 0
    total_samples = 0
    
    for intent_data in training_data:
        # Create or get intent
        intent, created = Intent.objects.get_or_create(
            name=intent_data['intent'],
            defaults={'description': intent_data['description']}
        )
        
        if created:
            print(f"✓ Created intent: {intent.name}")
            total_intents += 1
        else:
            print(f"→ Intent already exists: {intent.name}")
        
        # Add training samples
        for sample in intent_data['samples']:
            training, created = TrainingData.objects.get_or_create(
                intent=intent,
                question=sample['question'],
                defaults={'answer': sample['answer']}
            )
            
            if created:
                total_samples += 1
                print(f"  + Added: {sample['question'][:50]}...")
    
    print("\n" + "=" * 60)
    print("TRAINING DATA POPULATION COMPLETE!")
    print("=" * 60)
    print(f"Total Intents Created: {total_intents}")
    print(f"Total Training Samples Added: {total_samples}")
    print("\nYou can now train your model by visiting:")
    print("http://127.0.0.1:8000/admin-panel/")
    print("\nOr run: python manage.py shell")
    print(">>> from chatbot.ml_pipeline import ChatbotMLPipeline")
    print(">>> pipeline = ChatbotMLPipeline()")
    print(">>> pipeline.train()")
    print("=" * 60)

if __name__ == '__main__':
    populate_training_data()