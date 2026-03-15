"""
Hybrid Predictor - Category First with Natural Greetings/Goodbyes
Shows categories for information queries, direct responses for pleasantries
"""

import pickle
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .models import ModelVersion, TrainingData, Intent, Location
from .ml_hybridpipeline import HybridChatbotPipeline
from .entity_extractor import LocationExtractor

class HybridChatbotPredictor:
    """
    Smart predictor:
    - Direct answers: Greetings, Goodbyes (casual conversation)
    - Category display: Everything else (information queries)
    """
    
    _instance = None
    
    # STRICTER CONFIGURATION
    SIMILARITY_THRESHOLD = 0.35
    CONFIDENCE_THRESHOLD = 55
    
    def __init__(self):
        # Prevent reinitialization
        if not hasattr(self, "location_extractor"):
            self.location_extractor = LocationExtractor()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model_data = None
            cls._instance.load_model()
            cls._instance.location_extractor = LocationExtractor()
        return cls._instance
    
    def load_model(self):
        """Load the active hybrid model from database"""
        try:
            active_model = ModelVersion.objects.filter(is_active=True).latest('trained_at')
            
            if os.path.exists(active_model.model_path):
                with open(active_model.model_path, 'rb') as f:
                    self.model_data = pickle.load(f)
                print(f"✓ Loaded hybrid model version: {active_model.version}")
            else:
                print("⚠️  No active model found. Please train a model first.")
                self.model_data = None
        except ModelVersion.DoesNotExist:
            print("⚠️  No trained model found in database.")
            self.model_data = None
    
    def get_answer_from_database(self, intent_name):
        """Fetch the latest answer for an intent from database"""
        try:
            training_data = TrainingData.objects.filter(
                intent__name=intent_name,
                is_active=True
            ).first()
            
            if training_data:
                return training_data.answer
            else:
                return None
        except Exception as e:
            print(f"Error fetching answer: {e}")
            return None
    
    def get_intent_display_info(self, intent_name):
        """Get category and display name for an intent"""
        try:
            intent_obj = Intent.objects.filter(name=intent_name).first()
            if intent_obj:
                return {
                    'intent_name': intent_obj.name,
                    'name': intent_obj.name or intent_obj.name,
                    'category': intent_obj.category.name if intent_obj.category else 'General',
                    'category_display': intent_obj.category.name if intent_obj.category else 'General Information'
                }
            return None
        except Exception as e:
            print(f"Error getting intent info: {e}")
            return None
    
    def check_for_location_query(self, user_message):
        """
        Check if user is asking about a location/room
        Returns: (is_location_query, extracted_location, has_specific_room)
        """
        try:
            location_keywords = ['where', 'location', 'find', 'how to get', 
                                'directions', 'room', 'office', 'located']
            
            message_lower = user_message.lower()
            has_location_keyword = any(keyword in message_lower for keyword in location_keywords)
            
            if not has_location_keyword:
                return False, None, False
            
            # Try to extract specific location
            location = self.location_extractor.extract_location(user_message)
            
            if location:
                # Found specific location - show category with highlighting
                return True, location, True
            else:
                # General location query
                return True, None, False
                
        except Exception as e:
            print(f"[ERROR] Location check failed: {e}")
            return False, None, False
    
    def check_semantic_similarity(self, user_message):
        """
        Check if user message is semantically similar to ANY training question
        Returns: (max_similarity, most_similar_question)
        """
        try:
            user_embedding = self.model_data['sentence_encoder'].encode([user_message])
            
            similarities = cosine_similarity(
                user_embedding,
                self.model_data['training_embeddings']
            )[0]
            
            max_similarity_idx = np.argmax(similarities)
            max_similarity = float(similarities[max_similarity_idx])
            
            if 'training_questions' in self.model_data:
                most_similar_question = self.model_data['training_questions'][max_similarity_idx]
            else:
                most_similar_question = "Unknown"
            
            return max_similarity, most_similar_question
            
        except Exception as e:
            print(f"[ERROR] Semantic similarity check failed: {e}")
            return 0.0, ""
    
    def get_svm_prediction(self, user_message):
        """
        Get SVM intent prediction with confidence
        Returns: (predicted_intent, confidence)
        """
        try:
            pipeline = HybridChatbotPipeline()
            processed = pipeline.preprocess_text(user_message)
            
            vectorized = self.model_data['vectorizer'].transform([processed])
            
            prediction = self.model_data['svm_model'].predict(vectorized)[0]
            
            decision_scores = self.model_data['svm_model'].decision_function(vectorized)[0]
            
            max_score = np.max(np.abs(decision_scores))
            confidence = min(100, (max_score / (max_score + 0.5)) * 100)
            
            predicted_intent = self.model_data['reverse_mapping'][prediction]
            
            return predicted_intent, float(confidence)
            
        except Exception as e:
            print(f"[ERROR] SVM prediction failed: {e}")
            return 'unknown', 0.0
    
    def predict(self, user_message, request_type='initial'):
        """
        Smart prediction:
        - Greetings/Goodbyes: Direct response (natural conversation)
        - Locations: Category display (with highlighting if specific room)
        - Everything else: Category display
        
        Parameters:
        - user_message: The user's query
        - request_type: 'initial' or 'get_answer'
        """
        if not self.model_data:
            return {
                'response': "I'm not trained yet. Please contact the administrator.",
                'intent': None,
                'confidence': 0.0,
                'response_type': 'error'
            }
        
        user_lower = user_message.lower().strip()
        word_count = len(user_message.split())
        
        # ============================================
        # SPECIAL CASE 1: Greetings - DIRECT RESPONSE
        # ============================================
        greeting_keywords = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 
                            'good evening', 'greetings', 'howdy', 'sup', "what's up",
                            'hiya', 'yo', 'hola']
        
        if word_count <= 3:
            for keyword in greeting_keywords:
                if user_lower == keyword or user_lower.startswith(keyword + ' ') or user_lower.startswith(keyword + ','):
                    return {
                        'response': "Hello! Welcome to Lipa City Colleges. How can I help you today?",
                        'intent': 'greeting',
                        'confidence': 95.0,
                        'response_type': 'direct'  # Direct answer for greetings
                    }
        
        # ============================================
        # SPECIAL CASE 2: Goodbyes - DIRECT RESPONSE
        # ============================================
        goodbye_keywords = ['bye', 'goodbye', 'see you', 'thanks', 'thank you', 
                           'ok thanks', 'that\'s all', 'bye bye', 'thank you very much',
                           'thanks a lot', 'appreciate it']
        
        for keyword in goodbye_keywords:
            if user_lower == keyword or user_lower.startswith(keyword):
                return {
                    'response': "You're welcome! Have a great day! If you have more questions, feel free to ask anytime.",
                    'intent': 'goodbye',
                    'confidence': 95.0,
                    'response_type': 'direct'  # Direct answer for goodbyes
                }
        
        # ============================================
        # SPECIAL CASE 3: Locations - CATEGORY DISPLAY
        # ============================================
        is_location_query, location, has_specific_room = self.check_for_location_query(user_message)
        
        if is_location_query:
            # ALWAYS show location category (even if specific room found)
            # But highlight the specific room if found
            return {
                'response': None,
                'intent': 'ask_room_location',
                'category': 'locations',
                'category_display': 'Campus Locations',
                'confidence': 95.0,
                'response_type': 'location_category',
                'matched_location': location.room_number if location else None
            }
        
        # ============================================
        # NORMAL QUERIES - CATEGORY DISPLAY
        # ============================================
        semantic_similarity, most_similar = self.check_semantic_similarity(user_message)
        
        print(f"[DEBUG] Semantic: {semantic_similarity:.3f} | Similar to: '{most_similar}'")
        
        predicted_intent, svm_confidence = self.get_svm_prediction(user_message)
        
        print(f"[DEBUG] SVM: {svm_confidence:.1f}% | Intent: {predicted_intent}")
        
        # Decision thresholds
        semantic_pass = semantic_similarity >= self.SIMILARITY_THRESHOLD
        svm_pass = svm_confidence >= self.CONFIDENCE_THRESHOLD
        
        if not semantic_pass or not svm_pass:
            print(f"[DEBUG] REJECT - Semantic: {semantic_pass}, SVM: {svm_pass}")
            
            return {
                'response': (
                    "I'm not sure I understand your question. "
                    "Try to rephrase your question "
                    "or I'm not trained for that question yet."
                ),
                'intent': None,
                'confidence': min(semantic_similarity * 100, svm_confidence),
                'response_type': 'error'
            }
        
        print(f"[DEBUG] ACCEPT - Both checks passed")
        
        # Get intent display information
        intent_info = self.get_intent_display_info(predicted_intent)
        
        if not intent_info:
            return {
                'response': "I found a match but can't retrieve category information.",
                'intent': predicted_intent,
                'confidence': svm_confidence,
                'response_type': 'error'
            }
        
        # Show category for all information queries
        return {
            'response': None,
            'intent': predicted_intent,
            'intent_display': intent_info['name'],
            'category': intent_info['category'],
            'category_display': intent_info['category_display'],
            'confidence': svm_confidence,
            'response_type': 'category'
        }
    
    def get_answer_for_intent(self, intent_name):
        """
        Direct method to get answer for a specific intent
        Used when user clicks on a category item
        """
        try:
            answer = self.get_answer_from_database(intent_name)
            intent_info = self.get_intent_display_info(intent_name)
            
            if answer and intent_info:
                return {
                    'response': answer,
                    'intent': intent_name,
                    'intent_display': intent_info['name'],
                    'category': intent_info['category'],
                    'response_type': 'answer',
                    'confidence': 100.0
                }
            else:
                return {
                    'response': "Sorry, I couldn't retrieve the answer for this topic.",
                    'intent': intent_name,
                    'response_type': 'error',
                    'confidence': 0.0
                }
        except Exception as e:
            print(f"[ERROR] Failed to get answer: {e}")
            return {
                'response': "An error occurred while retrieving the answer.",
                'intent': intent_name,
                'response_type': 'error',
                'confidence': 0.0
            }
    
    def get_location_answer(self, room_number):
        """
        Get answer for a specific location/room
        Used when user clicks on a location in the location category
        """
        try:
            location = Location.objects.filter(
                room_number=room_number,
                is_active=True
            ).first()
            
            if location:
                response = self.location_extractor.get_location_response(location)
                return {
                    'response': response,
                    'intent': 'ask_room_location',
                    'response_type': 'answer',
                    'confidence': 100.0,
                    'entity': {
                        'type': 'location',
                        'value': location.room_number,
                        'name': location.room_name
                    }
                }
            else:
                return {
                    'response': f"Sorry, I couldn't find information about {room_number}.",
                    'intent': 'ask_room_location',
                    'response_type': 'error',
                    'confidence': 0.0
                }
        except Exception as e:
            print(f"[ERROR] Failed to get location: {e}")
            return {
                'response': "An error occurred while retrieving location information.",
                'intent': 'ask_room_location',
                'response_type': 'error',
                'confidence': 0.0
            }
    
    def reload_model(self):
        """Reload model and location extractor"""
        self.load_model()
        self.location_extractor = LocationExtractor()