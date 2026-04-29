"""
Hybrid Predictor - Direct Answer System with Entity Recognition
Gives immediate answers instead of showing categories
NOW WITH: Smart program entity recognition for "Do you offer IT?" queries

ROUTING PRIORITY:
  1. Greetings        → instant direct response
  2. Goodbyes         → instant direct response
  2.5 Out-of-scope / Profanity filter
        • Profanity            → polite warning (checked first, separate response)
        • Out-of-scope query   → redirect to LCC topics
        Runs AFTER greetings/goodbyes so "hi", "thanks" are never blocked.
        Runs BEFORE entity recognition and ML so garbage never reaches those layers.
  3. Entity Recognition → AVAILABILITY / CATEGORY queries only
                         ("Do you offer IT?", "Meron ba kayong Nursing?",
                          "Do you have medical programs?")
                         ← Informational queries ("What is Nursing?",
                            "Tell me about BSN") are intentionally skipped
                            here so ML training data answers them properly.
  4. Locations        → entity extraction
  5. ML Prediction    → semantic similarity + SVM (handles all informational
                         questions about specific programs)
"""

import pickle
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .models import ModelVersion, TrainingData, Intent, Location
from .ml_hybridpipeline import HybridChatbotPipeline
from .entity_extractor import LocationExtractor
from .program_matcher import ProgramEntityRecognizer


class HybridChatbotPredictor:
    """
    Direct answer predictor - no category menus.
    Entity recognition handles availability/category queries only;
    ML handles all informational program queries.
    """

    _instance = None

    # Configuration
    SIMILARITY_THRESHOLD = 0.35
    CONFIDENCE_THRESHOLD = 55

    # ── Out-of-scope detection ───────────────────────────────────────────────
    # If ANY scope keyword is present the message is considered in-scope and
    # the blacklist / prefix checks are skipped entirely (whitelist-first rule).
    SCOPE_KEYWORDS = [
        # School identity
        'lcc', 'lipa', 'college', 'university', 'school',
        # Programs (codes + names)
        'program', 'course', 'degree', 'major', 'curriculum', 'subject',
        'nursing', 'bsn', 'criminology', 'bscrim', 'crim',
        'computer science', 'bscs', 'computer engineering', 'bscpe',
        'business administration', 'bsba', 'accountancy', 'bsa', 'bsma',
        'elementary education', 'beed', 'secondary education', 'bsed',
        'psychology', 'psych', 'ab-psy',
        'hospitality', 'bshm', 'tourism', 'bstm',
        'education', 'it program', 'tech program',
        # Admissions / enrollment
        'admission', 'enroll', 'enrollment', 'apply', 'application',
        'freshman', 'requirements', 'document', 'credentials',
        # Financial
        'tuition', 'fee', 'scholarship', 'discount', 'financial aid',
        'payment', 'eba', 'ched', 'pagibig', 'pag-ibig', 'pwd',
        'pnp', 'solo parent', 'employee privilege', 'loyalty',
        # Campus
        'library', 'campus', 'room', 'office', 'building', 'floor', 'location',
        # School info
        'founder', 'founded', 'established', 'website', 'accreditation',
        # Tagalog school terms
        'kurso', 'mag-enroll', 'pag-aaral', 'iskolar', 'kolehiyo',
        'bayad', 'matrikula', 'pasukan', 'patungkol',
    ]

    # If ANY of these patterns match AND no scope keyword is found → out of scope.
    OUT_OF_SCOPE_PATTERNS = [
        # Weather
        'weather', 'forecast', 'temperature', 'rain today', 'typhoon alert',
        # Food / cooking (not hospitality program)
        'recipe', 'how to cook', 'ingredients for', 'cooking instructions',
        # Entertainment
        'movie', 'film', 'netflix', 'drama series', 'watch online',
        'song lyrics', 'music video', 'tiktok', 'instagram',
        # Sports scores (outside school athletics)
        'nba score', 'nfl game', 'pba score', 'fifa score',
        # Politics / government (unrelated to LCC)
        'president of the philippines', 'senator', 'election result',
        # Finance (unrelated to fees)
        'stock price', 'bitcoin', 'cryptocurrency', 'forex',
        # Personal / entertainment
        'tell me a joke', 'make me laugh', 'who is your crush',
        'relationship advice', 'love advice',
        # General homework/tasks
        'write my essay', 'do my homework', 'translate this sentence',
        'who invented', 'capital of',
    ]

    # Profanity / inappropriate words → always blocked regardless of scope keywords.
    # NOTE: These are checked in predict() BEFORE _is_out_of_scope() is called,
    # so they get their own distinct response. Do NOT check them again inside
    # _is_out_of_scope() — that would be dead code since predict() fires first.
    PROFANITY_WORDS = [
        # English
        'fuck', 'shit', 'bitch', 'asshole', 'bastard', 'damn you',
        'crap', 'dick', 'pussy', 'wtf', 'stfu',
        # Filipino
        'putangina', 'puta', 'gago', 'tangina', 'bobo',
        'ulol', 'tanga', 'leche', 'pakyu', 'punyeta',
        'hindot', 'kantot', 'tarantado', 'inutil',
    ]

    # "Who is X?" / "Do you know X?" prefixes — blocks random entity queries
    # when NO scope keyword is present (e.g. "who is batman?", "do you know spiderman?").
    #
    # IMPORTANT: 'what is' and 'tell me about' are also listed in
    # ProgramEntityRecognizer.ml_owned_patterns and are intentionally handled
    # by ML training data for LCC informational queries ("What is Nursing?",
    # "Tell me about BSN"). They are safe here ONLY because step 2 inside
    # _is_out_of_scope() (scope keyword check) fires first — any LCC-related
    # "what is X" or "tell me about X" query will always contain a SCOPE_KEYWORD
    # (e.g. 'nursing', 'bsn', 'program') and will never reach this prefix check.
    RANDOM_ENTITY_PREFIXES = [
        'who is', 'who was', 'who are',
        'do you know', 'have you heard of',
        'what is', 'tell me about',       # safe — only fires when no scope keyword found
        'sino si', 'ano si', 'kilala mo', # Tagalog equivalents
    ]
    # ────────────────────────────────────────────────────────────────────────

    def __init__(self):
        if not hasattr(self, 'location_extractor'):
            self.location_extractor = LocationExtractor()
            self.program_recognizer = ProgramEntityRecognizer()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model_data = None
            cls._instance.load_model()
            cls._instance.location_extractor = LocationExtractor()
            cls._instance.program_recognizer = ProgramEntityRecognizer()
            # Cache slot for semantic similarity — avoids computing it twice
            # when _is_out_of_scope() runs step 5 and CASE 5 both need it.
            cls._instance._similarity_cache = None
        return cls._instance

    def load_model(self):
        """Load the active hybrid model from database"""
        try:
            active_model = ModelVersion.objects.filter(is_active=True).latest('trained_at')

            if os.path.exists(active_model.model_path):
                with open(active_model.model_path, 'rb') as f:
                    self.model_data = pickle.load(f)
                print(f"[OK] Loaded hybrid model version: {active_model.version}")
            else:
                print("⚠️  No active model found. Please train a model first.")
                self.model_data = None
        except ModelVersion.DoesNotExist:
            print("⚠️  No trained model found in database.")
            self.model_data = None

    def get_answer_from_database(self, intent_name):
        """Fetch the answer for an intent from database"""
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

    def check_for_location_query(self, user_message):
        """
        Check if user is asking about a location/room.
        Returns: (is_location_query, extracted_location)
        """
        try:
            location_keywords = [
                'where', 'location', 'find','directions', 'room', 'office', 'located'
            ]

            message_lower = user_message.lower()
            has_location_keyword = any(keyword in message_lower for keyword in location_keywords)

            if not has_location_keyword:
                return False, None

            location = self.location_extractor.extract_location(user_message)
            return True, location

        except Exception as e:
            print(f"[ERROR] Location check failed: {e}")
            return False, None

    def check_semantic_similarity(self, user_message):
        """
        Check if user message is semantically similar to ANY training question.
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
        Get SVM intent prediction with confidence.
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

    def _is_out_of_scope(self, user_message):
        """
        Returns True when the message is clearly outside LCC's scope.

        NOTE: Profanity is intentionally NOT checked here. It is handled
        separately in predict() BEFORE this method is called, so that
        profanity gets its own distinct, kinder response. Adding a profanity
        check here would be dead code.

        Decision logic (short-circuit order):
          1. Single-word, non-scope      → blocked
          2. Any SCOPE_KEYWORD present   → in scope (return False immediately)
          3. Random entity prefix match  → blocked ("who is batman?")
          4. Explicit OUT_OF_SCOPE_PATTERN match → blocked
          5. Semantic similarity < 0.18  → blocked
             Result is stored in self._similarity_cache so CASE 5 in
             predict() can reuse it without a second encoder call.
          6. Default                     → let the ML pipeline decide
        """
        OUT_OF_SCOPE_SIMILARITY_CUTOFF = 0.18

        message_lower = user_message.lower().strip()
        words = message_lower.split()

        # ── 1. Single-word query that is not a scope keyword ─────────────────
        #    e.g. "batman", "lol", "asdf"
        #    (lone profanity is already caught before this method is called)
        if len(words) == 1 and words[0] not in self.SCOPE_KEYWORDS:
            print(f"[SCOPE] Single non-scope word: '{user_message}'")
            return True

        # ── 2. Scope keyword present → always in scope ───────────────────────
        #    This fires before steps 3-5 so no LCC-related query can ever be
        #    blocked by the blacklist or prefix checks below.
        if any(kw in message_lower for kw in self.SCOPE_KEYWORDS):
            return False

        # ── 3. Random entity prefix ("who is batman?", "do you know spiderman?")
        #    Only reaches here when NO scope keyword was found above.
        if any(
            message_lower.startswith(prefix) or f' {prefix} ' in message_lower
            for prefix in self.RANDOM_ENTITY_PREFIXES
        ):
            print(f"[SCOPE] Random entity query detected: '{user_message}'")
            return True

        # ── 4. Explicit out-of-scope pattern match ───────────────────────────
        if any(pattern in message_lower for pattern in self.OUT_OF_SCOPE_PATTERNS):
            print(f"[SCOPE] Out-of-scope pattern matched: '{user_message}'")
            return True

        # ── 5. Semantic similarity far below training data ───────────────────
        #    Acts as a last-resort catch for ambiguous queries that slipped
        #    through steps 1-4.
        #
        #    FIX: The result is cached in self._similarity_cache so that
        #    CASE 5 in predict() can reuse it directly without running the
        #    sentence encoder a second time for the same message.
        if self.model_data:
            similarity, closest = self.check_semantic_similarity(user_message)
            self._similarity_cache = (similarity, closest)  # store for CASE 5

            if similarity < OUT_OF_SCOPE_SIMILARITY_CUTOFF:
                print(
                    f"[SCOPE] Similarity {similarity:.3f} < {OUT_OF_SCOPE_SIMILARITY_CUTOFF} "
                    f"(closest: '{closest}') → out of scope"
                )
                return True

        return False  # Let the pipeline continue

    def predict(self, user_message):
        """
        DIRECT ANSWER prediction - no category menus.

        Priority Order:
          1. Greetings         → instant response
          2. Goodbyes          → instant response
          2.5 Profanity        → polite warning (checked before out-of-scope
                                 so it gets its own distinct response)
          2.5 Out-of-scope     → redirect to LCC topics
          3. Entity Recognition → availability / category queries ONLY
                                  Informational queries ("What is Nursing?") are
                                  intentionally skipped → handled by ML in CASE 5.
          4. Locations         → entity extraction
          5. ML Prediction     → semantic + SVM for all informational queries
                                  Reuses cached similarity from step 2.5 if
                                  _is_out_of_scope() already computed it.
        """
        if not self.model_data:
            return {
                'response': "I'm not trained yet. Please contact the administrator.",
                'intent': None,
                'confidence': 0.0,
                'response_type': 'error'
            }

        # Clear the similarity cache at the start of every new prediction.
        # This ensures stale values from a previous call are never reused.
        self._similarity_cache = None

        user_lower = user_message.lower().strip()
        word_count = len(user_message.split())

        # ============================================================
        # CASE 1: Greetings - DIRECT RESPONSE
        # ============================================================
        greeting_keywords = [
            'hi', 'hello', 'hey', 'good morning', 'good afternoon',
            'good evening', 'greetings', 'howdy', 'sup', "what's up",
            'hiya', 'yo', 'hola'
        ]

        if word_count <= 3:
            for keyword in greeting_keywords:
                if (user_lower == keyword
                        or user_lower.startswith(keyword + ' ')
                        or user_lower.startswith(keyword + ',')):
                    return {
                        'response': (
                            "I'm here to help answer your questions about our school! "
                            "Whether you're a prospective student, a parent, or just curious "
                            "about LCC, I'm happy to assist!\n\n"
                            "<strong>You can ask me about:</strong>\n"
                            "<strong>1.</strong> Our academic programs\n"
                            "<strong>2.</strong> Admission requirements and procedures\n"
                            "<strong>3.</strong> Tuition fees and scholarships\n"
                            "<strong>4.</strong> Enrollment schedules\n"
                            "<strong>5.</strong> Campus locations\n"
                            "<strong>6.</strong> Library hours\n"
                            "<strong></strong> ...and more!\n\n"
                            "What would you like to know?"
                        ),
                        'intent': 'greeting',
                        'confidence': 95.0,
                        'response_type': 'direct'
                    }

        # ============================================================
        # CASE 2: Goodbyes - DIRECT RESPONSE
        # ============================================================
        goodbye_keywords = [
            'bye', 'goodbye', 'see you', 'thanks', 'thank you',
            'ok thanks', "that's all", 'bye bye', 'thank you very much',
            'thanks a lot', 'appreciate it'
        ]

        for keyword in goodbye_keywords:
            if user_lower == keyword or user_lower.startswith(keyword):
                return {
                    'response': (
                        "If you have more questions later, don't hesitate to come back. "
                        "I'm always here! Whether it's about admissions, scholarships, or "
                        "anything about Lipa City Colleges, just ask!\n"
                        "Have a wonderful day, and I hope to see you as part of the LCC family soon!\n\n"
                        "Good luck with your college journey!"
                    ),
                    'intent': 'goodbye',
                    'confidence': 95.0,
                    'response_type': 'direct'
                }

        # ============================================================
        # CASE 2.5: OUT-OF-SCOPE / PROFANITY FILTER
        #
        # Runs AFTER greetings/goodbyes (always valid) but BEFORE entity
        # recognition and ML so unrelated or inappropriate queries never
        # reach those layers.
        #
        # Profanity is checked first and separately so it gets its own
        # distinct, kinder response rather than the generic out-of-scope
        # redirect. _is_out_of_scope() does NOT re-check profanity.
        #
        # Handles:
        #   • Profanity / inappropriate language  → polite warning
        #   • Single non-scope words              → out-of-scope redirect
        #   • Random entity queries ("who is batman?")
        #   • Explicit off-topic patterns (weather, movies, crypto …)
        #   • Semantically unrelated queries (similarity < 0.18)
        # ============================================================

        # ── Profanity: always blocked, distinct response ──────────────────────
        if any(word in user_lower for word in self.PROFANITY_WORDS):
            print(f"[SCOPE] Profanity blocked: '{user_message}'")
            return {
                'response': (
                    "I'm here to help with questions about "
                    "<strong>Lipa City Colleges (LCC)</strong>, "
                    "and I'd appreciate if we keep the conversation respectful. 😊\n\n"
                    "Feel free to ask me about admissions, programs, "
                    "scholarships, or anything else about LCC!"
                ),
                'intent': 'profanity',
                'confidence': 0.0,
                'response_type': 'out_of_scope'
            }

        # ── Out-of-scope: everything else that doesn't belong ─────────────────
        if self._is_out_of_scope(user_message):
            print(f"[SCOPE] Out-of-scope query blocked: '{user_message}'")
            return {
                'response': (
                    "I'm sorry, I can only answer questions about "
                    "<strong>Lipa City Colleges (LCC)</strong> "
                    "especially on admission and scholarship.\n\n"
                    "<strong>Here's what I can help you with:</strong>\n"
                    "<strong>1.</strong> Academic programs and courses\n"
                    "<strong>2.</strong> Admission requirements and procedures\n"
                    "<strong>3.</strong> Tuition fees and scholarships\n"
                    "<strong>4.</strong> Enrollment schedules\n"
                    "<strong>5.</strong> Campus locations and offices\n"
                    "<strong>6.</strong> Library hours and school information\n\n"
                    "Is there anything about LCC I can help you with?"
                ),
                'intent': 'out_of_scope',
                'confidence': 90.0,
                'response_type': 'out_of_scope'
            }

        # ============================================================
        # CASE 3: PROGRAM ENTITY RECOGNITION
        #
        # IMPORTANT — SCOPE IS LIMITED ON PURPOSE:
        #   ProgramEntityRecognizer.detect_program_query() now returns None
        #   for informational patterns like "what is", "tell me about",
        #   "ano ang", etc. Those fall through to CASE 5 (ML) so that your
        #   training data (about_nursing, about_criminology, …) answers them.
        #
        #   Entity recognition only fires for:
        #     • Availability queries  — "Do you offer IT?", "Meron ba kayong BSN?"
        #     • Category queries      — "IT related programs?", "Medical courses?"
        # ============================================================
        try:
            program_match = self.program_recognizer.detect_program_query(user_message)

            if program_match:
                match_type = program_match['type']

                # ── Program not offered (availability query, no match found)
                if match_type == 'not_found':
                    print(f"[ENTITY] Program not offered: '{user_message}'")
                    response_data = self.program_recognizer.generate_not_found_response(user_message)
                    return {
                        'response': response_data['response'],
                        'intent': response_data['intent'],
                        'confidence': response_data['confidence'],
                        'response_type': 'direct',
                        'method': 'entity_recognition'
                    }

                # ── Program match found (availability or category query)
                if match_type == 'program_match':
                    print(f"[ENTITY] Program entity detected: {len(program_match['programs'])} match(es)")
                    response_data = self.program_recognizer.generate_response(
                        program_match['programs'],
                        user_message
                    )
                    if response_data:
                        print(f"[ENTITY] Responding via entity recognition: {response_data['intent']}")
                        return {
                            'response': response_data['response'],
                            'intent': response_data['intent'],
                            'confidence': response_data['confidence'],
                            'response_type': 'direct',
                            'method': 'entity_recognition'
                        }

        except Exception as e:
            print(f"[ERROR] Entity recognition failed: {e}")
            # Continue to ML prediction

        # ============================================================
        # CASE 4: Locations - DIRECT ANSWER
        # ============================================================
        is_location_query, location = self.check_for_location_query(user_message)

        if is_location_query:
            if location:
                response = self.location_extractor.get_location_response(location)
                return {
                    'response': response,
                    'intent': 'ask_room_location',
                    'confidence': 95.0,
                    'response_type': 'direct',
                    'entity': {
                        'type': 'location',
                        'value': location.room_number,
                        'name': location.room_name
                    }
                }
            else:
                return {
                    'response': (
                        "I can help you find campus locations. "
                        "Please specify which office or room you're looking for."
                    ),
                    'intent': 'ask_room_location',
                    'confidence': 90.0,
                    'response_type': 'direct'
                }

        # ============================================================
        # CASE 5: Information Queries - ML PREDICTION
        #
        # All informational program queries land here:
        #   "What is Nursing?", "Tell me about BSN", "Ano ang Criminology?",
        #   "Jobs after Computer Science?", etc.
        #
        # FIX: Reuse the cached similarity if _is_out_of_scope() already
        # computed it during step 5 of the scope check. This avoids running
        # the sentence encoder twice for the same message.
        # ============================================================
        if self._similarity_cache is not None:
            semantic_similarity, most_similar = self._similarity_cache
            print(f"[DEBUG] Semantic (cached): {semantic_similarity:.3f} | Similar to: '{most_similar}'")
        else:
            semantic_similarity, most_similar = self.check_semantic_similarity(user_message)
            print(f"[DEBUG] Semantic: {semantic_similarity:.3f} | Similar to: '{most_similar}'")

        predicted_intent, svm_confidence = self.get_svm_prediction(user_message)
        print(f"[DEBUG] SVM: {svm_confidence:.1f}% | Intent: {predicted_intent}")

        semantic_pass = semantic_similarity >= self.SIMILARITY_THRESHOLD
        svm_pass      = svm_confidence      >= self.CONFIDENCE_THRESHOLD

        if not semantic_pass or not svm_pass:
            print(f"[DEBUG] REJECT - Semantic: {semantic_pass}, SVM: {svm_pass}")
            return {
                'response': (
                    "I'm not sure I understand your question. "
                    "Could you please rephrase it, or try asking in a different way? "
                    "I'm still learning!"
                ),
                'intent': None,
                'confidence': min(semantic_similarity * 100, svm_confidence),
                'response_type': 'error'
            }

        print(f"[DEBUG] ACCEPT - Both checks passed")

        answer = self.get_answer_from_database(predicted_intent)

        if not answer:
            return {
                'response': "I found a match but couldn't retrieve the answer. Please try again.",
                'intent': predicted_intent,
                'confidence': svm_confidence,
                'response_type': 'error'
            }

        return {
            'response': answer,
            'intent': predicted_intent,
            'confidence': svm_confidence,
            'response_type': 'direct'
        }

    def reload_model(self):
        """Reload model, location extractor, and program recognizer"""
        self.load_model()
        self.location_extractor = LocationExtractor()
        self.program_recognizer = ProgramEntityRecognizer()