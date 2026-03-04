"""
Hybrid ML Pipeline with Confidence Thresholds
Combines TF-IDF with Sentence Transformers for accuracy
"""

import pickle
import os
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
import re
from .models import TrainingData, ModelVersion, Intent

class HybridChatbotPipeline:
    """
    Hybrid approach combining:
    1. Sentence embeddings for semantic similarity
    2. TF-IDF + SVM for intent classification
    3. Confidence thresholds for reliability
    """
    
    def __init__(self):
        self.svm_model = None
        self.vectorizer = None
        self.intent_mapping = {}
        self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Training data for semantic similarity
        self.training_questions = []
        self.training_embeddings = None
        self.training_intents = []
        
    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        # Lowercase
        text = text.lower()
        
        # Expand common contractions
        contractions = {
            "what's": "what is", "where's": "where is", "how's": "how is",
            "when's": "when is", "it's": "it is", "i'm": "i am",
            "you're": "you are", "can't": "cannot", "won't": "will not",
            "don't": "do not", "isn't": "is not", "aren't": "are not",
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove punctuation
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def load_training_data(self):
        """Load data from database - now handles JSON questions"""
        data = TrainingData.objects.filter(is_active=True).select_related('intent')
        
        questions = []
        intents = []
        
        # Store for semantic similarity
        self.training_questions = []
        self.training_intents = []
        
        for item in data:
            # Get all questions for this intent
            question_list = item.get_questions()
            
            for question in question_list:
                preprocessed = self.preprocess_text(question)
                questions.append(preprocessed)
                intents.append(item.intent.name)
                
                # Store original for semantic similarity
                self.training_questions.append(question)
                self.training_intents.append(item.intent.name)
        
        return questions, intents
    
    def train(self):
        """Train the hybrid model"""
        print("="*70)
        print("HYBRID CHATBOT TRAINING")
        print("="*70)
        
        print("\n[1/6] Loading training data...")
        questions, intents = self.load_training_data()
        
        if len(questions) < 10:
            raise ValueError("Need at least 10 training samples to train the model")
        
        print(f"✓ Loaded {len(questions)} training samples")
        
        # Check distribution
        from collections import Counter
        intent_counts = Counter(intents)
        single_sample_intents = [intent for intent, count in intent_counts.items() if count == 1]
        
        if single_sample_intents:
            print(f"\n⚠️  WARNING: These intents have only 1 sample:")
            for intent in single_sample_intents:
                print(f"   - {intent}")
            print("Recommendation: Add at least 5 samples per intent for best results\n")
        
        # Create intent mapping
        unique_intents = list(set(intents))
        self.intent_mapping = {intent: idx for idx, intent in enumerate(unique_intents)}
        reverse_mapping = {idx: intent for intent, idx in self.intent_mapping.items()}
        
        labels = [self.intent_mapping[intent] for intent in intents]
        
        print(f"\n[2/6] Creating sentence embeddings (semantic understanding)...")
        # Create embeddings for ALL training questions
        self.training_embeddings = self.sentence_encoder.encode(
            self.training_questions,
            show_progress_bar=True
        )
        print("✓ Embeddings created")
        
        # Split data
        min_samples = min(intent_counts.values())
        
        if min_samples >= 2:
            print(f"\n[3/6] Splitting data (stratified)...")
            X_train, X_test, y_train, y_test = train_test_split(
                questions, labels, test_size=0.2, random_state=42, stratify=labels
            )
        else:
            print(f"\n[3/6] Splitting data (custom - some intents have 1 sample)...")
            intent_data = {}
            for q, l in zip(questions, labels):
                if l not in intent_data:
                    intent_data[l] = []
                intent_data[l].append(q)
            
            X_train, X_test, y_train, y_test = [], [], [], []
            
            for intent_id, samples in intent_data.items():
                if len(samples) == 1:
                    X_train.extend(samples)
                    y_train.extend([intent_id])
                else:
                    split_idx = max(1, int(len(samples) * 0.8))
                    X_train.extend(samples[:split_idx])
                    y_train.extend([intent_id] * split_idx)
                    X_test.extend(samples[split_idx:])
                    y_test.extend([intent_id] * (len(samples) - split_idx))
        
        print(f"✓ Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        print(f"\n[4/6] Vectorizing with enhanced TF-IDF...")
        self.vectorizer = TfidfVectorizer(
            max_features=2000,        # More vocabulary
            ngram_range=(1, 3),       # 1-3 word phrases
            min_df=1,                 # Keep rare words
            max_df=0.8,               # Ignore very common words
            sublinear_tf=True,        # Log scaling
        )
        
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        print("✓ Vectorization complete")
        
        print(f"\n[5/6] Training SVM classifier...")
        self.svm_model = LinearSVC(
            C=1.0,
            max_iter=10000,
            random_state=42
        )
        self.svm_model.fit(X_train_vec, y_train)
        print("✓ SVM training complete")
        
        print(f"\n[6/6] Evaluating model...")
        # Evaluate on test set
        if len(X_test) > 0:
            y_pred = self.svm_model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\n{'='*70}")
            print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
            print(f"{'='*70}")
            
            # Classification report
            test_intents = list(set(y_test))
            target_names = [reverse_mapping[i] for i in sorted(test_intents)]
            
            print("\nDetailed Performance by Intent:")
            print(classification_report(
                y_test, y_pred,
                labels=sorted(test_intents),
                target_names=target_names,
                zero_division=0
            ))
        else:
            # No test set - estimate
            y_train_pred = self.svm_model.predict(X_train_vec)
            accuracy = accuracy_score(y_train, y_train_pred)
            print(f"\n{'='*70}")
            print(f"Training Accuracy: {accuracy * 100:.2f}%")
            print("⚠️  No test set - add more samples for better evaluation")
            print(f"{'='*70}")
        
        # Save model
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = 'ml_models'
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f'hybrid_model_{version}.pkl')
        
        model_data = {
            'svm_model': self.svm_model,
            'vectorizer': self.vectorizer,
            'intent_mapping': self.intent_mapping,
            'reverse_mapping': reverse_mapping,
            'sentence_encoder': self.sentence_encoder,
            'test_questions': X_test,
            'test_intents': [reverse_mapping[i] for i in y_test],
            'training_questions': self.training_questions,
            'training_embeddings': self.training_embeddings,
            'training_intents': self.training_intents,
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save to database
        ModelVersion.objects.update(is_active=False)
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
            status = "🌟" if count >= 20 else "✅" if count >= 10 else "⚠️" if count >= 5 else "❌"
            print(f"  {status} {intent_name}: {count} samples")
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        
        return accuracy, version