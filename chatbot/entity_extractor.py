"""
Entity extraction for room/location queries
"""

import re
from .models import Location, LocationKeyword

class LocationExtractor:
    """
    Extracts room numbers and location names from user queries
    """
    
    def _is_actual_room_number(self, room_number):
        """Check if room_number should have "Room" prefix"""
        room_str = room_number.strip()
        
        if not room_str or len(room_str) > 10:
            return False
        
        if room_str.isdigit():              # "401" → True
            return True
        
        if room_str[0].isdigit() and len(room_str) <= 6:  # "3A" → True
            return True
        
        if all(c.isdigit() or c in '-/.' for c in room_str):  # "401-A" → True
            return True
        
        return False  # "CLINIC" → False
    
    def __init__(self):
        # Preload all locations and keywords for fast lookup
        self.locations = {}  # {room_number: Location}
        self.keyword_map = {}  # {keyword: Location}
        
        self._load_locations()
    
    def _load_locations(self):
        """Load all locations and keywords into memory"""
        locations = Location.objects.filter(is_active=True).prefetch_related('keywords')
        
        for location in locations:
            # Store by room number
            self.locations[location.room_number.lower()] = location
            
            # Store by keywords
            for keyword_obj in location.keywords.all():
                self.keyword_map[keyword_obj.keyword.lower()] = location
            
            # Store by aliases
            for alias in location.get_aliases():
                self.keyword_map[alias.lower()] = location
    
    def extract_location(self, text):
        """
        Extract location from text
        Returns: Location object or None
        """
        text_lower = text.lower()
        
        # Strategy 1: Direct room number match (e.g., "401", "402")
        room_patterns = [
            r'\broom\s*(\w+)',
            r'\br\s*(\d+)',
            r'#\s*(\w+)',
            r'\b(\d{3,4})\b',  # 3-4 digit numbers
        ]
        
        for pattern in room_patterns:
            match = re.search(pattern, text_lower)
            if match:
                room_num = match.group(1)
                if room_num in self.locations:
                    return self.locations[room_num]
        
        # Strategy 2: Keyword matching (longest match first)
        sorted_keywords = sorted(self.keyword_map.keys(), key=len, reverse=True)
        
        for keyword in sorted_keywords:
            if keyword in text_lower:
                return self.keyword_map[keyword]
        
        # Strategy 3: Partial matching for room names
        for room_num, location in self.locations.items():
            if location.room_name and location.room_name.lower() in text_lower:
                return location
        
        return None
    
    def get_location_response(self, location):
        if location.room_name:
            response = f"{location.room_name} is located at {location.floor}, {location.building}."
        else:
            room_num = location.room_number
            # NEW: Check if numeric before adding "Room"
            if self._is_actual_room_number(room_num):
                response = f"Room {room_num} is located at {location.floor}, {location.building}."
            else:
                response = f"{room_num} is located at {location.floor}, {location.building}."
        
        return response