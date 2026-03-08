import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'campusbot.settings')
django.setup()

from chatbot.models import Category, Intent, TrainingData

def populate_training_data():

    print("Populating academic training data (10 questions per intent)...")
    print("=" * 60)

    training_data = [

    # ===== ADMISSION REQUIREMENTS - FRESHMEN & TRANSFEREES =====
    {
        "category": "Admissions",
        "intent": "admission_requirements",
        "description": "Requirements for incoming freshmen and transferees",
        "questions": [
            "What are the admission requirements for freshmen?",
            "What documents do I need to apply as a freshman?",
            "Requirements for incoming freshmen students",
            "What should I submit for freshman admission?",
            "What are the needed documents for freshmen?",
            "What are the requirements for transferees?",
            "How can I transfer to this university?",
            "What documents are required for transfer students?",
            "Requirements for students transferring schools",
            "What do I need to submit as a transferee?"
        ],
        "answer": """
Incoming freshmen are required to submit the following documents:

• Form 138 (Report Card)
• Form 137 (Permanent Record)
• Certificate of Good Moral Character
• PSA Birth Certificate
• Three (3) copies of 2x2 ID photos
• Any other program-specific requirements
Applicants must also take the entrance examination and attend an admission interview.

Transferees must submit the following requirements:

• Transcript of Records (TOR)
• Certificate of Eligibility to Transfer (if applicable)
• Certificate of Good Moral Character
• PSA Birth Certificate
• Two or three 2x2 ID photos

The Registrar evaluates transferee TORs to determine creditable courses.
"""
    },

    # ===== ADMISSION PROCEDURE =====
    {
        "category": "Admissions",
        "intent": "admission_procedure",
        "description": "Steps in the admission process for new and transferee applicants",
        "questions": [
            "What is the admission procedure?",
            "How do I apply for admission?",
            "How do I get admitted to the school?",
            "What steps do I follow to apply?",
            "What should I do to get admitted?",
            "What are the steps for transferees to apply?",
            "How do I schedule an entrance exam?",
            "Where do I submit my requirements for admission?",
            "How long does the admission process take?",
            "Who can I contact about my admission status?"
        ],
        "answer": """
The admission process usually follows these steps:

NEW STUDENT
1. Fill out the online application form at the school's admissions portal.
2. Submit the required admission documents (see requirements).
3. Schedule and take the entrance examination.
4. Attend the admission interview at the Guidance & Admission Office.
5. Wait for admission results and receive the Admission Slip.
6. If accepted, proceed to enrollment (assessment and payment).

TRANSFEREE
1. Submit Transcript of Records and transfer credentials to the Registrar.
2. The Registrar evaluates coursework for credit transfer.
3. Attend interview/assessment if required by the department.
4. Receive notification of accepted credits and admission status.
"""
    },

    # ===== ENROLLMENT PROCEDURE =====
    {
        "category": "Enrollment",
        "intent": "enrollment_procedure",
        "description": "Steps for enrolling and registering for classes",
        "questions": [
            "How do I enroll in the university?",
            "What are the steps in enrollment?",
            "How can I register for classes?",
            "What is the enrollment procedure?",
            "What should I do during enrollment?",
            "How do I pay for enrollment?",
            "Where do I get my class schedule after enrollment?",
            "Can I enroll online and how?",
            "What documents should I bring when enrolling?",
            "How do returning students complete enrollment?"
        ],
        "answer": """
The enrollment process generally includes the following steps:

1. Secure the admission slip or approval for enrollment (new students).
2. Fill out the enrollment or pre-registration form.
3. Select or enlist subjects using the registration system or department enlistment.
4. Proceed to the Registrar for verification and encoding.
5. Pay the required tuition and miscellaneous fees at the Cashier/Accounting Office or via approved online channels.
6. Receive your official registration form and class schedule.
7. Activate your LMS account if applicable.
"""
    },

    # ===== GRADING SYSTEM =====
    {
        "category": "Academic Policies",
        "intent": "grading_system",
        "description": "Explanation of the grading system and computation",
        "questions": [
            "What is the grading system in the university?",
            "How are grades computed?",
            "What grades are considered passing?",
            "How does the grading system work?",
            "What does 1.00 or 3.00 mean in grading?",
            "How is my final grade calculated?",
            "What is the grading scale used by the school?",
            "Are there weightings for exams and assignments?",
            "What happens if I get a low grade?",
            "How can I request a grade review?"
        ],
        "answer": """
The grading system generally follows this scale:

1.00 – Excellent
1.25 – Very Good
1.50 – Good
1.75 – Satisfactory
2.00 – Fair
2.25 – Passing
2.50 – Conditional
3.00 – Lowest Passing Grade
5.00 – Failure

Grades are typically computed using weighted components (exams, quizzes, projects, attendance) as defined by each course syllabus. Final grade formulas vary by subject but are usually a weighted average of component grades. For grade reviews, follow the department's grade appeal policy.
"""
    },

    # ===== ATTENDANCE POLICY =====
    {
        "category": "Academic Policies",
        "intent": "attendance_policy",
        "description": "Rules regarding attendance, tardiness, and consequences",
        "questions": [
            "What is the attendance policy?",
            "How many absences are allowed?",
            "Can I fail due to absences?",
            "What happens if I miss many classes?",
            "Are students required to attend classes regularly?",
            "How do tardies affect attendance?",
            "What is the policy for long illness absences?",
            "When is a student dropped from a subject for absence?",
            "Do I need to submit documentation for absences?",
            "How does attendance affect my grade?"
        ],
        "answer": """
Students are expected to attend classes regularly.

Common rules include:
• Tardiness may be counted and converted to absences (e.g., 3 tardies = 1 absence).
• Excessive absences may lead to warnings, dropping from the subject, or failing the course.
• Students should submit valid documentation (e.g., medical certificate) for excused absences.
• Instructors may apply participation or attendance components to the final grade.
Refer to the Student Manual for exact thresholds and procedures.
"""
    },

    # ===== INCOMPLETE GRADE =====
    {
        "category": "Academic Policies",
        "intent": "incomplete_grade",
        "description": "Policy and steps to complete an INC (Incomplete) grade",
        "questions": [
            "What does INC mean?",
            "What is an incomplete grade?",
            "How do I complete an INC grade?",
            "What happens if I get an incomplete grade?",
            "How long do I have to complete an INC?",
            "Can I request an extension for an INC?",
            "Who do I contact about finishing an INC?",
            "What forms are needed to complete an INC grade?",
            "Will INC become a failing grade if not completed?",
            "How is the final grade recorded after completing an INC?"
        ],
        "answer": """
INC (Incomplete) is given when a student fails to complete course requirements within the term.

To complete an INC grade:
1. Coordinate with the course instructor to identify missing requirements.
2. Complete and submit the missing work within the allowed period.
3. Submit any required completion forms to the department.
4. Instructor submits the updated grade to the Registrar for encoding.

Failure to complete within the allowed timeframe may result in conversion to a failing grade. Check the Student Manual for the exact deadline and conversion rules.
"""
    },

    # ===== LIST OF ACADEMIC PROGRAMS =====
    {
    "category": "Academic Programs",
    "intent": "list_of_academic_programs",
    "description": "List of academic programs or courses offered by the university",
    "questions": [
        "What courses are offered in the university?",
        "What academic programs are available?",
        "What degrees can I take in your school?",
        "What programs does the university offer?",
        "What courses can I study here?",
        "Can you list the programs offered?",
        "Do you have Nursing or Criminology programs?",
        "What degree programs are in the College of Computing?",
        "Which business programs are available?",
        "Where can I see the list of courses offered?"
    ],
    "answer": """
The university offers the following academic programs:

College of Education and Liberal Arts
• Bachelor of Elementary Education (BEED) – Major in General Education
• Bachelor of Secondary Education (BSED) – Major in English, Mathematics, Filipino, and Science
• Bachelor of Arts in Psychology (AB-PSY)

College of Business and Accountancy
• Bachelor of Science in Business Administration (BSBA) – Major in Marketing, Human Resource Management, and Financial Management
• Bachelor of Science in Accountancy (BSA)
• Bachelor of Science in Management Accounting (BSMA)

College of Computing and Technology Engineering
• Bachelor of Science in Computer Science (BSCS)
• Bachelor of Science in Computer Engineering (BSCPE)

College of Nursing
• Bachelor of Science in Nursing (BSN)

College of Criminal Justice Education
• Bachelor of Science in Criminology (BSCRIM)

College of International Tourism and Hospitality Management
• Bachelor of Science in Hospitality Management (BSHM)
• Bachelor of Science in Tourism Management (BSTM)

For more information about a specific program, please ask about that program name.
"""
    },

    # ===== SCHOLARSHIP PROGRAMS =====
    {
    "category": "Scholarships",
    "intent": "scholarship_programs",
    "description": "Scholarship programs offered by the school including institutional and government scholarships",
    "questions": [
        "What scholarships are available in the school?",
        "Do you offer scholarships?",
        "What scholarship programs does the school have?",
        "Are there financial assistance or scholarships available?",
        "What are the scholarship opportunities for students?",
        "How do I apply for scholarships?",
        "What are the requirements for the academic scholarship?",
        "Are there government scholarships available?",
        "Who can I contact about scholarships?",
        "When are scholarship applications open?"
    ],
    "answer": """
Lipa City Colleges offers several scholarship programs to support deserving students.

INSTITUTIONAL SCHOLARSHIP PROGRAMS (examples)
• Academic Scholarship – 25% to 100% discount depending on GPA.
• Gawad Karunungan Scholarship – full scholarship for outstanding students.
• Carlos R. Mojares Scholarship Program – full tuition discount for student leaders.
• Family Privilege – discount for families with multiple enrollees.
• Solo Parent Discount – 20% discount for students with solo parents.
• Alumni Discount – 20% discount if parent is an alumnus.
• AYOUDA – 50% discount for students from selected distant areas.
• BEAP – 50% discount for students from Taal-affected areas.

NON-INSTITUTIONAL / GOVERNMENT SCHOLARSHIPS (examples)
• Mayor Eric B. Africa (EBA) Scholarship Program – local scholarship for qualifying residents.
• Provincial Capitol Scholarship Program – provincial financial assistance.
• CHED scholarship programs – national-level grants where applicable.

For application procedures, deadlines, and eligibility, contact the Student Affairs and Services Office or the Guidance & Admission Office.
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