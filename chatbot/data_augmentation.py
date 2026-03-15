"""
Bilingual Data Augmentation (English + Filipino)
Optimized for Lipa City Colleges Chatbot
"""

import random
from typing import List

try:
    import nlpaug.augmenter.word as naw
    NLPAUG_AVAILABLE = True
except ImportError:
    NLPAUG_AVAILABLE = False
    print("NLPAug not available - using simple augmentation only")


class BilingualAugmenter:
    """
    Handles both English and Filipino/Tagalog questions
    Optimized for college admission & scholarship queries
    """
    
    def __init__(self):
        # Initialize NLPAug if available
        if NLPAUG_AVAILABLE:
            self.synonym_aug = naw.SynonymAug(aug_src='wordnet')
        
        # English synonyms (college-specific)
        self.english_synonyms = {
            'scholarship': ['financial aid', 'grant', 'educational assistance'],
            'admission': ['enrollment', 'application', 'entry'],
            'requirements': ['needed documents', 'qualifications', 'prerequisites'],
            'apply': ['submit application', 'enroll', 'register'],
            'available': ['offered', 'provided', 'open', 'accessible'],
            'program': ['course', 'degree'],
            'fee': ['tuition', 'payment', 'cost'],
            'office': ['department', 'bureau'],
            'website': ['site', 'web page', 'online portal'],
            'contact': ['reach', 'get in touch with'],
            'documents': ['papers', 'files', 'credentials'],
            'student': ['enrollee', 'applicant'],
            'freshmen': ['freshman', 'first year', 'incoming student'],
            'transferee': ['transfer student', 'lateral entry student'],
        }
        
        # Filipino synonyms
        self.filipino_synonyms = {
            'ano': ['ano ang', 'anong'],
            'saan': ['saan ko', 'nasaan'],
            'paano': ['paano ko', 'papaano'],
            'kailan': ['kailan ang', 'kelan'],
            'magkano': ['magkano ang', 'presyo ng'],
            'may': ['mayroon', 'meron'],
            'ba': ['ba ang', 'ba yung'],
        }
        
        # LCC-specific abbreviations
        self.lcc_variants = {
            'LCC': ['Lipa City Colleges', 'the school', 'the university'],
            'Lipa City Colleges': ['LCC', 'the school'],
        }
    
    def detect_language(self, text: str) -> str:
        """Detect if text is English or Filipino"""
        filipino_markers = ['ano', 'saan', 'paano', 'ba', 'ng', 'ang', 'mga', 'sa', 'ko', 'mo']
        text_lower = text.lower()
        
        filipino_count = sum(1 for marker in filipino_markers if marker in text_lower.split())
        
        return 'filipino' if filipino_count >= 2 else 'english'
    
    def synonym_replacement(self, text: str, language: str) -> List[str]:
        """Replace words with synonyms based on language"""
        augmented = []
        words = text.split()
        
        # Choose appropriate synonym dictionary
        synonyms = self.filipino_synonyms if language == 'filipino' else self.english_synonyms
        
        for _ in range(2):  # Generate 2 variations
            new_words = words.copy()
            replaced = False
            
            for i, word in enumerate(new_words):
                clean_word = word.lower().strip('?.!,')
                
                # Try LCC variants first
                for key, variants in self.lcc_variants.items():
                    if clean_word == key.lower():
                        new_words[i] = random.choice(variants)
                        replaced = True
                        break
                
                # Try synonym replacement
                if clean_word in synonyms and random.random() > 0.5:
                    new_words[i] = random.choice(synonyms[clean_word])
                    replaced = True
            
            if replaced:
                augmented.append(' '.join(new_words))
        
        return augmented
    
    def make_variations(self, text: str, language: str) -> List[str]:
        """Create length variations"""
        variations = []
        
        if language == 'english':
            # Shorter version (remove fillers)
            fillers = ['please', 'can you', 'could you', 'I want to know', 'tell me']
            shorter = text.lower()
            for filler in fillers:
                shorter = shorter.replace(filler, '').strip()
            shorter = ' '.join(shorter.split())  # Clean extra spaces
            
            if shorter and shorter != text.lower():
                variations.append(shorter)
            
            # Add politeness if missing
            if not any(word in text.lower() for word in ['please', 'can', 'could']):
                # Remove question mark, add polite prefix
                base = text.strip('?').strip()
                variations.append(f"Can you tell me {base.lower()}?")
        
        else:  # Filipino
            # Shorter version
            if 'ba' in text.lower():
                shorter = text.replace('ba', '').replace('  ', ' ').strip()
                variations.append(shorter)
            
            # Longer version (add 'po' for politeness)
            if 'po' not in text.lower():
                variations.append(text.replace('?', ' po?'))
        
        return variations
    
    def question_restructuring(self, text: str, language: str) -> List[str]:
        """Restructure questions"""
        augmented = []
        
        if language == 'english':
            text_lower = text.lower().strip()
            
            # Detect question word
            q_words = {
                'what': ['Can you tell me what', 'I want to know what', 'What exactly'],
                'where': ['Can you tell me where', 'I need to find where', 'Where exactly'],
                'how': ['Can you explain how', 'Steps to', 'Process to'],
                'when': ['Can you tell me when', 'What time'],
                'who': ['Can you tell me who', 'I need to know who'],
            }
            
            for q_word, templates in q_words.items():
                if text_lower.startswith(q_word):
                    rest = text_lower.split(' ', 1)[1] if ' ' in text_lower else ''
                    rest = rest.strip('?')
                    
                    # Apply 1-2 templates
                    for template in random.sample(templates, min(2, len(templates))):
                        new_q = f"{template} {rest}?"
                        augmented.append(new_q.capitalize())
                    break
        
        return augmented
    
    def augment_question(self, question: str, num_augmentations: int = 4) -> List[str]:
        """
        Main augmentation method
        Returns: [original] + augmented versions
        """
        all_augmented = [question]  # Always keep original
        
        # Detect language
        language = self.detect_language(question)
        
        # 1. Synonym replacement
        all_augmented.extend(self.synonym_replacement(question, language))
        
        # 2. Variations (shorter/longer)
        all_augmented.extend(self.make_variations(question, language))
        
        # 3. Question restructuring (English only)
        if language == 'english':
            all_augmented.extend(self.question_restructuring(question, language))
        
        # 4. NLPAug (English only)
        if language == 'english' and NLPAUG_AVAILABLE:
            try:
                aug = self.synonym_aug.augment(question)
                if isinstance(aug, list):
                    all_augmented.extend(aug)
                else:
                    all_augmented.append(aug)
            except:
                pass
        
        # Remove duplicates
        unique = []
        seen = set()
        for aug in all_augmented:
            aug_clean = aug.lower().strip()
            if aug_clean and aug_clean not in seen:
                seen.add(aug_clean)
                unique.append(aug)
        
        # Return up to num_augmentations + 1 (original)
        return unique[:num_augmentations + 1]