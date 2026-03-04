import os
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'campusbot.settings')
django.setup()

from chatbot.models import Category


def populate_categories():
    """Populate default chatbot categories"""

    print("Populating category database...")
    print("=" * 60)

    categories_data = [
        {
            'name': 'Enrollment',
            'icon': '📝',
            'description': 'Enrollment, registration, and admission',
            'order': 1
        },
        {
            'name': 'Schedules',
            'icon': '📅',
            'description': 'Class schedules, exam dates, and academic calendar',
            'order': 2
        },
        {
            'name': 'Scholarships',
            'icon': '💰',
            'description': 'Scholarships, grants, and financial aid',
            'order': 3
        },
        {
            'name': 'Locations',
            'icon': '🗺️',
            'description': 'Find rooms, buildings, and facilities',
            'order': 4
        },
        {
            'name': 'Facilities',
            'icon': '🏢',
            'description': 'Library, labs, canteen, and other facilities',
            'order': 5
        },
        {
            'name': 'Requirements',
            'icon': '📋',
            'description': 'Documents, forms, and requirements',
            'order': 6
        },
        {
            'name': 'Fees & Payments',
            'icon': '💵',
            'description': 'Tuition, fees, and payment information',
            'order': 7
        },
        {
            'name': 'Policies',
            'icon': '📜',
            'description': 'Rules, regulations, and guidelines',
            'order': 8
        },
    ]

    created_count = 0

    for data in categories_data:

        # Create or update category
        category, created = Category.objects.update_or_create(
            name=data['name'],
            defaults={
                'icon': data['icon'],
                'description': data['description'],
                'order': data['order']
            }
        )

        if created:
            print(f"✓ Created: {category}")
            created_count += 1
        else:
            print(f"→ Updated: {category}")

        # Preview
        preview = f'  Preview: "{category.name} - {category.description}"'
        print(preview)

    print("\n" + "=" * 60)
    print(f"✓ Created {created_count} new categories")
    print(f"✓ Total categories: {Category.objects.count()}")
    print("=" * 60)


if __name__ == "__main__":
    populate_categories()