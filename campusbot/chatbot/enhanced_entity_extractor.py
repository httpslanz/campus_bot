"""
Enhanced Location Extractor - Comprehensive Matching
Searches across: room_number, room_name, keywords, and aliases
"""

import re
from django.db.models import Q
from .models import Location, LocationKeyword


class EnhancedLocationExtractor:
    """
    Enhanced location extraction with comprehensive matching:
    - Room numbers (exact and partial)
    - Room names (case-insensitive, fuzzy)
    - Keywords from LocationKeyword model
    - Aliases
    """
    
    def __init__(self):
        self._load_locations()
    
    def _load_locations(self):
        """Load all active locations with their keywords and aliases"""
        self.locations = Location.objects.filter(is_active=True).prefetch_related('keywords')
        
        # Build comprehensive search index
        self.search_index = []
        
        for location in self.locations:
            # Create entry with all searchable terms
            entry = {
                'location': location,
                'searchable_terms': self._build_searchable_terms(location)
            }
            self.search_index.append(entry)
    
    def _build_searchable_terms(self, location):
        """
        Build list of all searchable terms for a location
        Returns: list of (term, priority, match_type)
        """
        terms = []
        
        # 1. Room number - HIGHEST priority
        if location.room_number:
            terms.append({
                'term': location.room_number.lower().strip(),
                'priority': 100,
                'match_type': 'room_number'
            })
        
        # 2. Room name - HIGH priority
        if location.room_name:
            terms.append({
                'term': location.room_name.lower().strip(),
                'priority': 90,
                'match_type': 'room_name'
            })
            # Add individual words from room name
            for word in location.room_name.split():
                if len(word) > 2:  # Skip very short words
                    terms.append({
                        'term': word.lower().strip(),
                        'priority': 85,
                        'match_type': 'room_name_word'
                    })
        
        # 3. Keywords - Based on keyword priority
        for keyword in location.keywords.all():
            terms.append({
                'term': keyword.keyword.lower().strip(),
                'priority': 70 + keyword.priority,  # Use keyword priority
                'match_type': 'keyword'
            })
        
        # 4. Aliases - MEDIUM priority
        aliases = location.get_aliases()
        for alias in aliases:
            terms.append({
                'term': alias.lower().strip(),
                'priority': 80,
                'match_type': 'alias'
            })
            # Add individual words from aliases
            for word in alias.split():
                if len(word) > 2:
                    terms.append({
                        'term': word.lower().strip(),
                        'priority': 75,
                        'match_type': 'alias_word'
                    })
        
        # 5. Building + Floor combination - LOW priority
        building_floor = f"{location.building} {location.floor}".lower()
        terms.append({
            'term': building_floor,
            'priority': 60,
            'match_type': 'building_floor'
        })
        
        return terms
    
    def extract_location(self, user_message):
        """
        Extract location from user message with comprehensive matching
        Returns: Location object or None
        """
        if not user_message:
            return None
        
        message_lower = user_message.lower().strip()
        
        # Remove common question words to improve matching
        clean_message = self._clean_message(message_lower)
        
        # Find all potential matches
        matches = []
        
        for entry in self.search_index:
            location = entry['location']
            searchable_terms = entry['searchable_terms']
            
            # Check each searchable term
            for term_info in searchable_terms:
                term = term_info['term']
                priority = term_info['priority']
                match_type = term_info['match_type']
                
                # Calculate match score
                score = self._calculate_match_score(
                    clean_message, 
                    term, 
                    priority,
                    match_type
                )
                
                if score > 0:
                    matches.append({
                        'location': location,
                        'score': score,
                        'matched_term': term,
                        'match_type': match_type
                    })
        
        # Return best match if score is high enough
        if matches:
            # Sort by score descending
            matches.sort(key=lambda x: x['score'], reverse=True)
            best_match = matches[0]
            
            # Require minimum score threshold
            if best_match['score'] >= 50:
                print(f"[LOCATION MATCH] Found: {best_match['location'].room_number} "
                      f"(score: {best_match['score']}, "
                      f"matched: '{best_match['matched_term']}', "
                      f"type: {best_match['match_type']})")
                return best_match['location']
        
        return None
    
    def _clean_message(self, message):
        """Remove common question words to improve matching"""
        # Remove question words
        question_words = [
            'where', 'is', 'the', 'what', 'location', 'of', 'find', 
            'how', 'to', 'get', 'directions', 'room', 'office', 
            'located', 'at', 'in', 'can', 'you', 'tell', 'me',
            'show', 'about', 'please', 'help', '?', 'a', 'an'
        ]
        
        words = message.split()
        cleaned_words = [w for w in words if w not in question_words]
        
        return ' '.join(cleaned_words)
    
    def _calculate_match_score(self, message, term, base_priority, match_type):
        """
        Calculate match score based on how well term matches message
        Returns: score (0-100+)
        """
        if not term or not message:
            return 0
        
        score = 0
        
        # EXACT MATCH - Highest score
        if term == message:
            score = base_priority + 50
        
        # EXACT MATCH as whole word
        elif f" {term} " in f" {message} ":
            score = base_priority + 40
        
        # STARTS WITH
        elif message.startswith(term):
            score = base_priority + 30
        
        # ENDS WITH
        elif message.endswith(term):
            score = base_priority + 25
        
        # CONTAINS (substring)
        elif term in message:
            score = base_priority + 20
        
        # WORD MATCH (any word in message matches term)
        elif term in message.split():
            score = base_priority + 35
        
        # PARTIAL WORD MATCH
        else:
            # Check if any word in message partially matches term
            for word in message.split():
                if term in word or word in term:
                    score = base_priority + 10
                    break
        
        # BONUS: Give extra points for room_number matches
        if match_type == 'room_number' and score > 0:
            score += 20
        
        # BONUS: Give extra points for exact room_name matches
        if match_type == 'room_name' and score >= base_priority + 40:
            score += 15
        
        return score
    
    def get_location_response(self, location):
        """
        Generate a friendly response with location details
        """
        response = f"📍 **{location.room_name or location.room_number}**\n\n"
        
        if location.room_name and location.room_number:
            response += f"**Room Number:** {location.room_number}\n"
        
        response += f"**Building:** {location.building}\n"
        response += f"**Floor:** {location.floor}\n"
        
        if location.description:
            response += f"\n**Description:** {location.description}\n"
        
        # Add aliases if available
        aliases = location.get_aliases()
        if aliases:
            response += f"\n**Also known as:** {', '.join(aliases)}\n"
        
        return response
    
    def search_locations_by_query(self, query, limit=10):
        """
        Search for locations matching a query
        Returns: List of (location, score, matched_term) tuples
        """
        if not query:
            return []
        
        message_lower = query.lower().strip()
        clean_message = self._clean_message(message_lower)
        
        matches = []
        
        for entry in self.search_index:
            location = entry['location']
            searchable_terms = entry['searchable_terms']
            
            best_score = 0
            best_term = None
            best_type = None
            
            for term_info in searchable_terms:
                score = self._calculate_match_score(
                    clean_message,
                    term_info['term'],
                    term_info['priority'],
                    term_info['match_type']
                )
                
                if score > best_score:
                    best_score = score
                    best_term = term_info['term']
                    best_type = term_info['match_type']
            
            if best_score > 0:
                matches.append({
                    'location': location,
                    'score': best_score,
                    'matched_term': best_term,
                    'match_type': best_type
                })
        
        # Sort by score descending
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top results
        return matches[:limit]
    
    def get_all_locations_grouped(self):
        """
        Get all locations grouped by building
        Returns: dict of {building: [locations]}
        """
        grouped = {}
        
        for location in self.locations:
            building = location.building
            if building not in grouped:
                grouped[building] = []
            grouped[building].append(location)
        
        return grouped


# Backward compatibility alias
LocationExtractor = EnhancedLocationExtractor