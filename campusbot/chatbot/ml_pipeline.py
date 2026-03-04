import pickle
import os
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
from .models import TrainingData, ModelVersion, Intent

class ChatbotMLPipeline:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.intent_mapping = {}
        
    def preprocess_text(self, text):
        """Clean and normalize text"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def load_training_data(self):
        """Load data from database"""
        data = TrainingData.objects.filter(is_active=True).select_related('intent')
        
        questions = []
        intents = []
        
        for item in data:
            questions.append(self.preprocess_text(item.question))
            intents.append(item.intent.name)
        
        return questions, intents
    
    def train(self):
        """Train the model with current database data"""
        print("Loading training data...")
        questions, intents = self.load_training_data()
        
        if len(questions) < 10:
            raise ValueError("Need at least 10 training samples to train the model")
        
        # Check for intents with only 1 sample
        from collections import Counter
        intent_counts = Counter(intents)
        single_sample_intents = [intent for intent, count in intent_counts.items() if count == 1]
        
        if single_sample_intents:
            print("\n⚠️  WARNING: The following intents have only 1 sample:")
            for intent in single_sample_intents:
                print(f"   - {intent}")
            print("\nPlease add at least 2 samples per intent for better training.")
            print("Intents with only 1 sample will be excluded from test set.\n")
        
        # Create intent to ID mapping
        unique_intents = list(set(intents))
        self.intent_mapping = {intent: idx for idx, intent in enumerate(unique_intents)}
        reverse_mapping = {idx: intent for intent, idx in self.intent_mapping.items()}
        
        # Convert intents to numeric labels
        labels = [self.intent_mapping[intent] for intent in intents]
        
        # MODIFIED: Smart train-test split
        # Check if we can stratify
        min_samples = min(intent_counts.values())
        
        if min_samples >= 2:
            # Safe to use stratified split
            print(f"Using stratified split (minimum {min_samples} samples per intent)")
            X_train, X_test, y_train, y_test = train_test_split(
                questions, labels, test_size=0.2, random_state=42, stratify=labels
            )
        else:
            # Cannot stratify - use simple split or custom approach
            print("Using custom split due to intents with single samples...")
            
            # Separate data by intent
            intent_data = {}
            for q, l in zip(questions, labels):
                if l not in intent_data:
                    intent_data[l] = []
                intent_data[l].append(q)
            
            X_train = []
            X_test = []
            y_train = []
            y_test = []
            
            for intent_id, samples in intent_data.items():
                if len(samples) == 1:
                    # Put single sample in training only
                    X_train.extend(samples)
                    y_train.extend([intent_id])
                else:
                    # Split normally
                    split_idx = max(1, int(len(samples) * 0.8))
                    X_train.extend(samples[:split_idx])
                    y_train.extend([intent_id] * split_idx)
                    X_test.extend(samples[split_idx:])
                    y_test.extend([intent_id] * (len(samples) - split_idx))
            
            print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        print("Vectorizing text...")
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print("Training model...")
        self.model = MultinomialNB()
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate only if we have test data
        if len(X_test) > 0:
            y_pred = self.model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\n{'='*60}")
            print(f"Model Accuracy: {accuracy * 100:.2f}%")
            print(f"{'='*60}")
            
            # Only show classification report for intents in test set
            test_intents = list(set(y_test))
            target_names = [reverse_mapping[i] for i in sorted(test_intents)]
            
            print("\nClassification Report (Test Set Only):")
            print(classification_report(y_test, y_pred, 
                                       labels=sorted(test_intents),
                                       target_names=target_names,
                                       zero_division=0))
        else:
            # No test set - use training accuracy as approximation
            y_train_pred = self.model.predict(X_train_vec)
            accuracy = accuracy_score(y_train, y_train_pred)
            print(f"\n{'='*60}")
            print(f"Training Accuracy: {accuracy * 100:.2f}%")
            print("⚠️  No separate test set - add more samples for better evaluation")
            print(f"{'='*60}")
        
        # Save model
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = 'ml_models'
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f'chatbot_model_{version}.pkl')
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'intent_mapping': self.intent_mapping,
            'reverse_mapping': reverse_mapping,
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save to database
        ModelVersion.objects.update(is_active=False)  # Deactivate old models
        ModelVersion.objects.create(
            version=version,
            model_path=model_path,
            accuracy=accuracy * 100,
            is_active=True,
            training_samples=len(questions)
        )
        
        print(f"\n✓ Model saved: {model_path}")
        print(f"✓ Database record created")
        print(f"\nTotal training samples: {len(questions)}")
        print(f"Total intents: {len(unique_intents)}")
        
        # Show intent distribution
        print("\nIntent Distribution:")
        for intent_name, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
            status = "⚠️ " if count == 1 else "✓ "
            print(f"  {status}{intent_name}: {count} samples")
        
        return accuracy, version