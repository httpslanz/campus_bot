import os
import django
import json

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "campusbot.settings")
django.setup()

from chatbot.models import Location, LocationKeyword


def create_location(room_number, room_name, building, floor):

    location, created = Location.objects.update_or_create(
        room_number=room_number,
        defaults={
            "room_name": room_name,
            "building": building,
            "floor": floor,
            "description": "",
            "is_active": True
        }
    )

    location.set_aliases([])
    location.save()

    # Remove old keywords
    LocationKeyword.objects.filter(location=location).delete()

    keywords = [
        f"room {room_number}",
        room_number
    ]

    # Named rooms
    if not room_number.isdigit():
        keywords.append(room_number.lower())

    # Save keywords
    for key in set(keywords):
        LocationKeyword.objects.create(
            location=location,
            keyword=key.lower(),
            priority=1
        )

    print(f"✓ {'Created' if created else 'Updated'}: Room {room_number} ({floor})")


def expand_rooms(room):

    # Expand: 201-203 → [201,202,203]
    if "-" in room and room.replace("-", "").isdigit():
        start, end = room.split("-")
        return [str(i) for i in range(int(start), int(end) + 1)]

    return [room]


def populate_locations():

    print("Populating RBB1 Locations...")
    print("=" * 60)

    building = "RBB1"

    locations = [


        ("201-203", "Second Floor"),

        ("301-312", "Third Floor"),

        # -------- FOURTH FLOOR --------
        ("401-405", "Fourth Floor"),


        # -------- FIFTH FLOOR --------
        ("501-506", "Fifth Floor"),

    ]

    for room, floor in locations:

        expanded = expand_rooms(room)

        for single_room in expanded:
            create_location(
                room_number=single_room,
                room_name="",
                building=building,
                floor=floor
            )

    print("=" * 60)
    print(f"✓ Total Locations: {Location.objects.count()}")
    print("=" * 60)


if __name__ == "__main__":
    populate_locations()