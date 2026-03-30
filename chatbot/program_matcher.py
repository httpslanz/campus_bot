# PROGRAM ENTITY RECOGNITION MODULE
# Dynamic program matching without separate training data for each variation
# Add this to your Django app as: program_matcher.py
#
# FIX: Entity recognition now only handles:
#   1. Availability queries  → "Do you offer IT?", "Meron ba kayong Nursing?"
#   2. Category queries      → "IT related programs?", "Do you have medical courses?"
#   3. Not-found queries     → "Do you offer Medicine?" → "Sorry, we don't offer that"
#
# Informational queries ("What is Nursing?", "Tell me about BSN") are intentionally
# excluded here so they fall through to ML, which has proper training data for them.

class ProgramEntityRecognizer:
    """
    Recognizes program names and categories from user queries dynamically.
    No need for separate training data - uses entity matching!

    SCOPE: Availability and category queries ONLY.
    Informational queries are handled by the ML pipeline (training data exists).
    """

    def __init__(self):
        self.programs = {
            'BSN': {
                'full_name': 'Bachelor of Science in Nursing',
                'short_names': ['BSN', 'Nursing', 'BS Nursing', 'BS in Nursing'],
                'keywords': ['nurse', 'nursing', 'medical', 'health', 'healthcare', 'hospital', 'patient care'],
                'category': ['medical', 'health', 'healthcare', 'allied health'],
                'description': 'BS Nursing (BSN) - 4-year healthcare program',
                'college': 'College of Nursing'
            },
            'BSCRIM': {
                'full_name': 'Bachelor of Science in Criminology',
                'short_names': ['BSCRIM', 'Criminology', 'BS Criminology', 'Crim'],
                'keywords': ['criminology', 'police', 'law enforcement', 'investigation', 'crim', 'pnp'],
                'category': ['law enforcement', 'criminal justice', 'police'],
                'description': 'BS Criminology (BSCRIM) - 4-year law enforcement program',
                'college': 'College of Criminal Justice Education'
            },
            'BSCS': {
                'full_name': 'Bachelor of Science in Computer Science',
                'short_names': ['BSCS', 'Computer Science', 'BS Computer Science', 'CS', 'CompSci'],
                'keywords': ['computer science', 'programming', 'coding', 'software', 'IT', 'tech', 'technology', 'comsci'],
                'category': ['IT', 'technology', 'computer', 'tech'],
                'description': 'BS Computer Science (BSCS) - 4-year software development program',
                'college': 'College of Computing and Technology Engineering'
            },
            'BSCPE': {
                'full_name': 'Bachelor of Science in Computer Engineering',
                'short_names': ['BSCPE', 'Computer Engineering', 'BS Computer Engineering', 'CPE', 'CompEng'],
                'keywords': ['computer engineering', 'hardware', 'electronics', 'embedded', 'IT', 'tech', 'technology', 'comeng'],
                'category': ['IT', 'technology', 'computer', 'engineering', 'tech'],
                'description': 'BS Computer Engineering (BSCPE) - 4-year hardware+software program',
                'college': 'College of Computing and Technology Engineering'
            },
            'BSBA': {
                'full_name': 'Bachelor of Science in Business Administration',
                'short_names': ['BSBA', 'Business Administration', 'BS Business Admin', 'Business Admin'],
                'keywords': ['business', 'management', 'marketing', 'hr', 'human resource', 'finance', 'financial management'],
                'category': ['business', 'commerce', 'management'],
                'description': 'BS Business Administration (BSBA) - Majors: Marketing, HRM, Financial Management',
                'college': 'College of Business and Accountancy'
            },
            'BSA': {
                'full_name': 'Bachelor of Science in Accountancy',
                'short_names': ['BSA', 'Accountancy', 'BS Accountancy', 'Accounting'],
                'keywords': ['accountancy', 'accounting', 'cpa', 'auditing', 'finance', 'business'],
                'category': ['business', 'accounting', 'finance'],
                'description': 'BS Accountancy (BSA) - 4-year CPA preparation program',
                'college': 'College of Business and Accountancy'
            },
            'BSMA': {
                'full_name': 'Bachelor of Science in Management Accounting',
                'short_names': ['BSMA', 'Management Accounting', 'BS Management Accounting'],
                'keywords': ['management accounting', 'accounting', 'business', 'finance'],
                'category': ['business', 'accounting', 'finance'],
                'description': 'BS Management Accounting (BSMA) - Combines accounting and business management',
                'college': 'College of Business and Accountancy'
            },
            'BEED': {
                'full_name': 'Bachelor of Elementary Education',
                'short_names': ['BEED', 'Elementary Education', 'BS Elementary Education', 'Elem Ed'],
                'keywords': ['elementary education', 'teaching', 'teacher', 'education', 'elem ed'],
                'category': ['education', 'teaching'],
                'description': 'Bachelor of Elementary Education (BEED) - Teach grades 1-6',
                'college': 'College of Education and Liberal Arts'
            },
            'BSED': {
                'full_name': 'Bachelor of Secondary Education',
                'short_names': ['BSED', 'Secondary Education', 'BS Secondary Education', 'Sec Ed'],
                'keywords': ['secondary education', 'teaching', 'teacher', 'education', 'high school', 'sec ed'],
                'category': ['education', 'teaching'],
                'description': 'Bachelor of Secondary Education (BSED) - Majors: English, Math, Filipino, Science',
                'college': 'College of Education and Liberal Arts'
            },
            'AB-PSY': {
                'full_name': 'Bachelor of Arts in Psychology',
                'short_names': ['AB-PSY', 'AB Psychology', 'Psychology', 'Psych', 'AB Psych'],
                'keywords': ['psychology', 'psych', 'behavior', 'counseling'],
                'category': ['social science', 'liberal arts', 'psychology'],
                'description': 'AB Psychology (AB-PSY) - 4-year human behavior program',
                'college': 'College of Education and Liberal Arts'
            },
            'BSHM': {
                'full_name': 'Bachelor of Science in Hospitality Management',
                'short_names': ['BSHM', 'Hospitality Management', 'BS Hospitality', 'HRM', 'Hotel Management'],
                'keywords': ['hospitality', 'hotel', 'restaurant', 'food', 'culinary', 'tourism', 'hrm'],
                'category': ['hospitality', 'tourism', 'service'],
                'description': 'BS Hospitality Management (BSHM) - Hotels, restaurants, events',
                'college': 'College of International Tourism and Hospitality Management'
            },
            'BSTM': {
                'full_name': 'Bachelor of Science in Tourism Management',
                'short_names': ['BSTM', 'Tourism Management', 'BS Tourism', 'Tourism'],
                'keywords': ['tourism', 'travel', 'tour', 'hospitality'],
                'category': ['tourism', 'hospitality', 'travel'],
                'description': 'BS Tourism Management (BSTM) - Travel and tourism industry',
                'college': 'College of International Tourism and Hospitality Management'
            }
        }

        # ─────────────────────────────────────────────────────────────────────
        # Informational phrases that the ML training data already covers.
        # If the user's message contains these, skip entity recognition entirely.
        # ─────────────────────────────────────────────────────────────────────
        self.ml_owned_patterns = [
            'what is', 'what are',
            'tell me about',
            'ano ang', 'anong',
            'explain',
            'how long is',
            'what will i study',
            'jobs after',
            'career in',
            'subjects in',
            'why study',
            'what majors',
            'program details',
            'course info',
            'program like',
            'course like',
        ]

        # ─────────────────────────────────────────────────────────────────────
        # Availability / existence phrases — entity recognition handles these.
        # ─────────────────────────────────────────────────────────────────────
        self.availability_patterns = [
            'do you offer', 'do you have',
            'may kayo', 'mayroon ba', 'meron ba', 'meron kayong',
            'available ba', 'available ang',
            'nag-offer', 'nag-aalok',
            'is there a', 'is there an',
            'are there',
        ]

        # ─────────────────────────────────────────────────────────────────────
        # Category / group phrases — entity recognition handles these too.
        # ─────────────────────────────────────────────────────────────────────
        self.category_patterns = [
            'related program', 'related course', 'related programs', 'related courses',
            'program about', 'course about',
            'programs related', 'courses related',
            'programs in', 'courses in',
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def detect_program_query(self, user_message):
        """
        Detect if user is asking about program availability or a program category.

        Returns:
          {'type': 'program_match', 'programs': [...], ...}  — matched programs found
          {'type': 'not_found', 'query': ...}                — availability query but
                                                               no matching program
          None                                               — not an availability/
                                                               category query; let ML handle

        KEY RULE:
          - Informational queries ("What is Nursing?") → None → ML handles it
          - Availability queries  ("Do you offer Nursing?") → program_match or not_found
          - Category queries      ("IT related programs?")  → program_match or not_found
        """
        message_lower = user_message.lower()

        # Step 1: If it matches an ML-owned pattern, bail out immediately.
        if any(pattern in message_lower for pattern in self.ml_owned_patterns):
            print(f"[ENTITY] Skipping — ML-owned pattern in: '{user_message}'")
            return None

        # Step 2: Check whether it is an availability or category query.
        is_availability = any(pattern in message_lower for pattern in self.availability_patterns)
        is_category     = any(pattern in message_lower for pattern in self.category_patterns)

        if not (is_availability or is_category):
            return None

        # Step 3: Try to match specific programs or categories.
        matched_programs = self.match_programs(message_lower)

        if matched_programs:
            return {
                'type': 'program_match',
                'programs': matched_programs,
                'query': user_message,
                'is_availability': is_availability,
                'is_category': is_category,
            }

        # Step 4: It was an availability/category query but nothing matched.
        # Signal "not found" so the predictor replies with a proper message
        # instead of confusingly falling through to ML.
        print(f"[ENTITY] Availability query with no program match: '{user_message}'")
        return {
            'type': 'not_found',
            'query': user_message,
        }

    def match_programs(self, message_lower):
        """
        Match programs based on name, keywords, or category.
        """
        matched = []

        for code, info in self.programs.items():
            # Exact / short-name matches
            for name in info['short_names']:
                if name.lower() in message_lower:
                    if code not in [m['code'] for m in matched]:
                        matched.append({'code': code, 'info': info, 'match_type': 'exact'})
                    break

            # Keyword matches
            for keyword in info['keywords']:
                if keyword in message_lower:
                    if code not in [m['code'] for m in matched]:
                        matched.append({'code': code, 'info': info, 'match_type': 'keyword'})
                    break

            # Category matches
            for category in info['category']:
                if category in message_lower:
                    if code not in [m['code'] for m in matched]:
                        matched.append({'code': code, 'info': info, 'match_type': 'category'})
                    break

        return matched if matched else None

    def generate_response(self, matched_programs, query):
        """
        Generate a conversational response for matched programs.
        Only called for availability / category queries.
        """
        if not matched_programs:
            return None

        if len(matched_programs) == 1:
            return self._single_program_response(matched_programs[0], query)
        else:
            return self._multiple_programs_response(matched_programs, query)

    def generate_not_found_response(self, query):
        """
        Response when the user asks about a program LCC doesn't offer.
        e.g. "Do you offer Medicine?", "Meron ba kayong Law?",
             "Is there an Architecture program?"

        Lists all available programs so the user can pick an alternative.
        """
        # Build a grouped list of all offered programs
        by_college = {}
        for code, info in self.programs.items():
            college = info['college']
            if college not in by_college:
                by_college[college] = []
            by_college[college].append(f"{info['full_name']} ({code})")

        all_programs_list = ""
        for college, programs in by_college.items():
            all_programs_list += f"**{college}:**\n"
            for p in programs:
                all_programs_list += f"• {p}\n"
            all_programs_list += "\n"

        return {
            'response': (
                "I'm sorry, but that program is not currently offered at Lipa City Colleges.\n\n"

                "<strong>AVAILABLE PROGRAMS:</strong>\n\n"
                + "".join(
                    f"<strong>{college}:</strong>\n" +
                    "".join(
                        f"<strong>{i+1}.</strong> {program}\n"
                        for i, program in enumerate(programs)
                    ) +
                    "\n"
                    for college, programs in by_college.items()
                ) +
                "Would you like to know more about any of these programs? Just ask and I will be happy to help."
            ),
            'intent': 'program_not_offered',
            'confidence': 95.0,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _single_program_response(self, program, query):
        """
        Response for a single-program availability query.
        e.g. "Do you offer BSN?" / "Meron ba kayong Nursing?"
        """
        info = program['info']
        code = program['code']

        return {
            'response': (
                f"Yes, we offer <strong>{info['full_name']} ({code})</strong>.\n\n"

                f"<strong>PROGRAM:</strong> {info['description']}\n"
                f"<strong>COLLEGE:</strong> {info['college']}\n"
                f"<strong>DURATION:</strong> 4 years\n\n"

                "<strong>You can ask about:</strong>\n"
                "<strong>1.</strong> Curriculum and subjects\n"
                "<strong>2.</strong> Career opportunities\n"
                "<strong>3.</strong> Admission requirements\n"
                "<strong>4.</strong> Tuition fees\n\n"

                "Just ask and I will be happy to help."
            ),
            'intent': f'program_entity_single_{code}',
            'confidence': 95.0
        }

    def _multiple_programs_response(self, programs, query):
        """
        Response for category queries.
        e.g. "Do you have IT programs?", "Meron ba kayong medical courses?"
        """
        category_detected = self._detect_category(query.lower())

        response = f"Great question! Yes, we offer several <strong>{category_detected}</strong> programs.\n\n"

        by_college = {}
        for prog in programs:
            college = prog['info']['college']
            if college not in by_college:
                by_college[college] = []
            by_college[college].append(prog)

        for college, progs in by_college.items():
            response += f"<strong>{college}:</strong>\n"
            
            counter = 1
            for prog in progs:
                response += f"<strong>{counter}.</strong> {prog['info']['full_name']} ({prog['code']})\n"
                counter += 1
            
            response += "\n"

        # Use first matched program as a concrete footer example
        example_code = programs[0]['code']
        example_name = programs[0]['info']['full_name']

        response += (
            "<strong>PROGRAM DURATION:</strong> All programs are 4 years (8 semesters)\n\n"
            "Want to know more about any specific program? Just ask:\n"
            f"<strong>1.</strong> Tell me about {example_name}\n"
            f"<strong>2.</strong> What is {example_code}\n"
            f"<strong>3.</strong> Career opportunities in {example_code}\n\n"
            "I'm here to help."
        )

        return {
            'response': response,
            'intent': f'program_entity_category_{category_detected}',
            'confidence': 90.0
        }

    def _detect_category(self, message_lower):
        """
        Detect which category the user is asking about.
        """
        categories = {
            'IT':              ['it', 'computer', 'technology', 'tech', 'programming', 'software'],
            'medical':         ['medical', 'health', 'healthcare', 'nursing', 'nurse'],
            'business':        ['business', 'commerce', 'management', 'accounting'],
            'education':       ['education', 'teaching', 'teacher'],
            'law enforcement': ['law', 'police', 'criminology', 'enforcement'],
            'hospitality':     ['hospitality', 'tourism', 'hotel', 'travel'],
            'engineering':     ['engineering', 'engineer'],
        }

        for category, keywords in categories.items():
            if any(kw in message_lower for kw in keywords):
                return category

        return "related"