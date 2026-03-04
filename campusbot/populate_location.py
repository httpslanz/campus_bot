import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'campusbot.settings')
django.setup()

from chatbot.models import Location, LocationKeyword

def populate_locations():
    """Populate sample room/location data"""
    
    print("Populating location database...")
    print("=" * 60)
    
    locations_data = [
        {
            'room_number': '401',
            'room_name': 'Computer Science Lab',
            'building': 'Building A',
            'floor': '4th Floor',
            'description': 'Computer laboratory with 40 workstations',
            'keywords': ['401', 'cs lab', 'computer science lab', 'comp sci lab', 'computer lab']
        },
        {
            'room_number': '402',
            'room_name': 'Physics Lab',
            'building': 'Building A',
            'floor': '4th Floor',
            'description': 'Physics laboratory with experiment equipment',
            'keywords': ['402', 'physics lab', 'physics laboratory']
        },
        {
            'room_number': '403',
            'room_name': 'Chemistry Lab',
            'building': 'Building A',
            'floor': '4th Floor',
            'description': 'Chemistry laboratory',
            'keywords': ['403', 'chemistry lab', 'chem lab']
        },
        {
            'room_number': '202',
            'room_name': 'Lecture Hall B',
            'building': 'Building A',
            'floor': '2nd Floor',
            'description': 'Large lecture hall, capacity 150 students',
            'keywords': ['202', 'lecture hall b', 'lhb', 'lh-b']
        },
        {
            'room_number': 'Smart Lab',
            'room_name': '',  # Will use room_number in response
            'building': 'Building C',
            'floor': '1st Floor',
            'description': 'IoT and AI research laboratory',
            'keywords': ['smart lab', 'smartlab', 'smart technology lab', 'iot lab', 'ai lab']
        },
        {
            'room_number': 'Library',
            'room_name': '',
            'building': 'Building B',
            'floor': 'All Floors',
            'description': 'Campus library with study areas and resources',
            'keywords': ['library', 'main library', 'campus library']
        },
        {
            'room_number': 'Registrar',
            'room_name': "Registrar's Office",
            'building': 'Administration Building',
            'floor': 'Ground Floor',
            'description': 'Student registration and records office',
            'keywords': ['registrar', 'registration office', 'registrars office', 'records office']
        },
        {
            'room_number': 'Cafeteria',
            'room_name': '',
            'building': 'Student Center',
            'floor': 'Ground Floor',
            'description': 'Main dining facility',
            'keywords': ['cafeteria', 'dining hall', 'cafe', 'food court', 'canteen']
        }
    ]
    
    created_count = 0
    
    for data in locations_data:
        # Create or update location
        location, created = Location.objects.update_or_create(
            room_number=data['room_number'],
            defaults={
                'room_name': data['room_name'],
                'building': data['building'],
                'floor': data['floor'],
                'description': data['description'],
            }
        )
        
        if created:
            print(f"✓ Created: {location}")
            created_count += 1
        else:
            print(f"→ Updated: {location}")
        
        # Create keywords
        LocationKeyword.objects.filter(location=location).delete()  # Clear old
        
        for keyword in data['keywords']:
            LocationKeyword.objects.create(
                location=location,
                keyword=keyword.lower(),
                priority=1
            )
        
        print(f"  Added {len(data['keywords'])} keywords")
        
        # Show preview response
        if location.room_name:
            preview = f'  Response: "{location.room_name} is located at {location.floor}, {location.building}."'
        else:
            preview = f'  Response: "Room {location.room_number} is located at {location.floor}, {location.building}."'
        print(preview)
    
    print("\n" + "=" * 60)
    print(f"✓ Created {created_count} new locations")
    print(f"✓ Total locations: {Location.objects.count()}")
    print("=" * 60)

if __name__ == '__main__':
    populate_locations()