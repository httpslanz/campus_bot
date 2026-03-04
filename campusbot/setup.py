# setup_office.py
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'campusbot.settings')
django.setup()

from chatbot.models import User, Office, OfficeUser

def setup_office_staff():
    # Create offices
    offices_data = [
        {'name': 'Registrar Office', 'description': 'Student registration', 'email': 'registrar@campus.edu'},
        {'name': 'Library', 'description': 'Campus library services', 'email': 'library@campus.edu'},
        {'name': 'Financial Aid', 'description': 'Student financial services', 'email': 'finaid@campus.edu'},
    ]
    
    for office_data in offices_data:
        office, created = Office.objects.get_or_create(
            name=office_data['name'],
            defaults={
                'description': office_data['description'],
                'contact_email': office_data['email'],
                'is_active': True
            }
        )
        if created:
            print(f"✓ Created office: {office.name}")
        
        # Create a staff user for this office
        username = office_data['name'].lower().replace(' ', '_') + '_staff'
        user, created = User.objects.get_or_create(
            username=username,
            defaults={
                'email': office_data['email'],
                'first_name': office_data['name'],
                'last_name': 'Staff'
            }
        )
        if created:
            user.set_password('password123')  # Change this!
            user.save()
            print(f"✓ Created user: {username} (password: password123)")
        
        # Link user to office
        office_user, created = OfficeUser.objects.get_or_create(
            user=user,
            defaults={
                'office': office,
                'role': 'staff'
            }
        )
        if created:
            print(f"✓ Linked {username} to {office.name}")
    
    print("\n" + "="*50)
    print("Setup Complete!")
    print("="*50)
    print("\nYou can now login at: http://127.0.0.1:8000/login/")
    print("\nOffice Staff Accounts:")
    print("-" * 50)
    for office in Office.objects.all():
        staff = office.staff.first()
        if staff:
            print(f"Office: {office.name}")
            print(f"  Username: {staff.user.username}")
            print(f"  Password: password123")
            print()

if __name__ == '__main__':
    setup_office_staff()