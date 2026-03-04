import os
import django
import json

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "campusbot.settings")
django.setup()

from chatbot.models import Location, LocationKeyword


def create_location(room_number, room_name, building, floor, description="", aliases=None):

    if aliases is None:
        aliases = []

    location, created = Location.objects.update_or_create(
        room_number=room_number,
        defaults={
            "room_name": room_name,
            "building": building,
            "floor": floor,
            "description": description,
            "is_active": True
        }
    )

    # Save aliases
    location.set_aliases(aliases)
    location.save()

    # Remove old keywords
    LocationKeyword.objects.filter(location=location).delete()

    keywords = []

    # Numeric rooms
    if room_number.replace("-", "").isdigit():
        if "-" in room_number:
            start, end = room_number.split("-")
            for i in range(int(start), int(end) + 1):
                keywords.append(f"room {i}")
                keywords.append(str(i))
        else:
            keywords.append(f"room {room_number}")
            keywords.append(room_number)

    # Named locations
    else:
        keywords.append(room_number.lower())
        if room_name:
            keywords.append(room_name.lower())

    # Save keywords
    for key in set(keywords):
        LocationKeyword.objects.create(
            location=location,
            keyword=key.strip().lower(),
            priority=1
        )

    print(f"✓ {'Created' if created else 'Updated'}: {room_number} ({floor})")


def populate_locations():

    print("Populating RBB1 Locations...")
    print("=" * 60)

    building = "RBB1"

    locations = [

        # ================= GROUND FLOOR =================
        ("Chapel", "", "Ground Floor"),
        ("Canteen", "", "Ground Floor"),
        ("College of Education and Liberal Arts Office", "", "Ground Floor"),
        ("Consultation Room (GF)", "", "Ground Floor"),
        ("Generator Area", "", "Ground Floor"),
        ("Main Clinic", "", "Ground Floor"),
        ("General Services Office", "", "Ground Floor"),
        ("CSG / Vision Office", "", "Ground Floor"),
        ("College of Nursing Office", "", "Ground Floor"),
        ("Laboratory Custodian Office", "", "Ground Floor"),
        ("Biology Laboratory", "", "Ground Floor"),
        ("Physics Laboratory", "", "Ground Floor"),
        ("Bullet Recovery Room", "", "Ground Floor"),
        ("Student Services Office", "", "Ground Floor"),
        ("Culture and Sports Office", "", "Ground Floor"),
        ("Community Extension Office", "", "Ground Floor"),
        ("Testing Room", "", "Ground Floor"),
        ("Admissions and Guidance Office", "", "Ground Floor"),

        # ================= SECOND FLOOR =================
        ("Administration Office", "", "Second Floor"),
        ("Accreditation Room", "", "Second Floor"),
        ("Records Office", "", "Second Floor"),
        ("Accounting Office", "", "Second Floor"),
        ("Academic Affairs Office", "", "Second Floor"),
        ("Career Placement Office", "", "Second Floor"),
        ("Human Resource Office", "", "Second Floor"),
        ("Consultation Room (2F)", "", "Second Floor"),
        ("Research and Graduate School Office", "", "Second Floor"),
        ("Multimedia Room", "", "Second Floor"),
        ("Printing Office", "", "Second Floor"),
        ("Speech Laboratory", "", "Second Floor"),
        ("201-203", "", "Second Floor"),
        ("College of Criminology Office", "", "Second Floor"),
        ("Crime Laboratory", "", "Second Floor"),
        ("Chemistry Laboratory", "", "Second Floor"),
        ("Crime Scene Room", "", "Second Floor"),
        ("Criminology Consultation Room", "", "Second Floor"),
        ("Moot Court Room", "", "Second Floor"),
        ("Dark Room", "", "Second Floor"),
        ("Defense Tactics Room", "", "Second Floor"),
        ("Smart Laboratory", "", "Second Floor"),
        ("Management Information Office", "", "Second Floor"),

        # ================= THIRD FLOOR =================
        ("College Library", "", "Third Floor"),
        ("301-312", "", "Third Floor"),
        ("College of Business and Accountancy Office", "", "Third Floor"),
        ("Psychology Laboratory", "", "Third Floor"),
        ("Tagaytay Area", "", "Third Floor"),

        # ================= FOURTH FLOOR =================
        ("College of International Hospitality and Tourism Management Office", "", "Fourth Floor"),
        ("Mock Hotel", "", "Fourth Floor"),
        ("Function Room", "", "Fourth Floor"),
        ("Kitchen Art Laboratory", "", "Fourth Floor"),
        ("401-405", "", "Fourth Floor"),
        ("Drawing Laboratory", "", "Fourth Floor"),
        ("TESDA Laboratory", "", "Fourth Floor"),
        ("Computer Laboratory A", "", "Fourth Floor"),
        ("College of Computer Studies Office", "", "Fourth Floor"),
        ("College of Engineering Office", "", "Fourth Floor"),
        ("Computer Laboratory B", "", "Fourth Floor"),
        ("Criminology Review Room", "", "Fourth Floor"),

        # ================= FIFTH FLOOR =================
        ("Audio Visual Room", "", "Fifth Floor"),
        ("Mock Hotel Rooms", "", "Fifth Floor"),
        ("501-506", "", "Fifth Floor"),
        ("Mock Travel Agency Office", "", "Fifth Floor"),
    ]

    for room, name, floor in locations:
        create_location(
            room_number=room,
            room_name=name,
            building=building,
            floor=floor
        )

    print("=" * 60)
    print(f"✓ Total Locations: {Location.objects.count()}")
    print("=" * 60)


if __name__ == "__main__":
    populate_locations()