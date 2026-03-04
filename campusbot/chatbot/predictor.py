import pickle
import os
from .models import ModelVersion, TrainingData, Intent, IntentResponse
from .ml_pipeline import ChatbotMLPipeline

class ChatbotPredictor:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model_data = None
            cls._instance.load_model()
        return cls._instance
    
    def load_model(self):
        """Load the active model from database"""
        try:
            active_model = ModelVersion.objects.filter(is_active=True).latest('trained_at')
            
            if os.path.exists(active_model.model_path):
                with open(active_model.model_path, 'rb') as f:
                    self.model_data = pickle.load(f)
                print(f"Loaded model version: {active_model.version}")
            else:
                print("No active model found. Please train a model first.")
                self.model_data = None
        except ModelVersion.DoesNotExist:
            print("No trained model found in database.")
            self.model_data = None
    
    def get_answer_from_database(self, intent_name):
        """Fetch answer from IntentResponse table - SUPER DYNAMIC!"""
        try:
            intent = Intent.objects.get(name=intent_name)
            
            # Get the highest priority active response
            response = IntentResponse.objects.filter(
                intent=intent,
                is_active=True
            ).first()  # Already ordered by priority
            
            if response:
                return response.answer
            else:
                # Fallback to TrainingData if no IntentResponse exists
                training_data = TrainingData.objects.filter(
                    intent=intent,
                    is_active=True
                ).first()
                return training_data.answer if training_data else "I don't have information about that yet."
                
        except Intent.DoesNotExist:
            return "I don't have information about that yet."
        except Exception as e:
            print(f"Error fetching answer: {e}")
            return "Sorry, I encountered an error."
    
    def predict(self, user_message):
        """Predict intent and get response"""
        if not self.model_data:
            return {
                'response': "I'm not trained yet. Please contact the administrator.",
                'intent': None,
                'confidence': 0.0
            }
        
        # Preprocess
        pipeline = ChatbotMLPipeline()
        processed = pipeline.preprocess_text(user_message)
        
        # Vectorize
        vectorized = self.model_data['vectorizer'].transform([processed])
        
        # Predict intent
        prediction = self.model_data['model'].predict(vectorized)[0]
        probabilities = self.model_data['model'].predict_proba(vectorized)[0]
        confidence = max(probabilities) * 100
        
        # Get intent name
        predicted_intent = self.model_data['reverse_mapping'][prediction]
        
        # FETCH ANSWER DYNAMICALLY FROM DATABASE
        answer = self.get_answer_from_database(predicted_intent)
        
        return {
            'response': answer,
            'intent': predicted_intent,
            'confidence': confidence
        }
    
    def reload_model(self):
        """Reload model after retraining"""
        self.load_model()