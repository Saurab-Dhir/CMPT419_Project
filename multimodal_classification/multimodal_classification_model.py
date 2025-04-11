class Evidence():
    def __init__(self, emotion: str, confidence: float = 1.0, reliability: float = 1.0):
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        if not 0.0 <= reliability <= 1.0:
            raise ValueError("Reliability must be between 0 and 1")
        self.emotion = emotion
        self.confidence = confidence
        self.reliability = reliability

class MultiModalClassifier():
    def __init__(self):
        self.__E = {'angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad'}

        # Maps emotions to tense categories
        self.__POSITIVE_EMOTION = 1
        self.__NEGATIVE_EMOTION = -1
        self.__NEUTRAL_EMOTION = 0
        self.__EMOTION_LIB = {
            'happy': self.__POSITIVE_EMOTION,
            'angry': self.__NEGATIVE_EMOTION,
            'disgust': self.__NEGATIVE_EMOTION,
            'fearful': self.__NEGATIVE_EMOTION,
            'sad': self.__NEGATIVE_EMOTION,
            'neutral': self.__NEUTRAL_EMOTION
        }

    def mass_function(self, evidence):
        if evidence.emotion not in self.__E:
            return { frozenset(self.__E): 1.0 }
        return {
            frozenset([evidence.emotion]): evidence.confidence,
            frozenset(self.__E): 1 - evidence.confidence
        }

    def predict(self, tone, face, semantics):
        """
        args:
            tone: Evidence
            face: Evidence
            semantics: Evidence
        """
        E = ('angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad')
        
        # Define mass functions for each modality
        m_tone = self.mass_function(tone)
        m_face = self.mass_function(face)
        m_semantics = self.mass_function(semantics)

        # Discount evidences based classifier reliability
        m_tone = self.__discount_mass_function(m_tone, tone.reliability)
        m_face = self.__discount_mass_function(m_face, face.reliability)
        m_semantics = self.__discount_mass_function(m_semantics, semantics.reliability)
        
        # Discount semantics when nonverbal and verbal evidences are incongruent
        INCONGRUENCE_DISCOUNT = 0.5
        if self.__incongruent(tone, face, semantics):
            m_semantics = self.__discount_mass_function(m_semantics, INCONGRUENCE_DISCOUNT)
        
        # Fuse mass functions
        m_fused = self.__combine_mass_functions(m_tone, m_face)
        m_fused = self.__combine_mass_functions(m_fused, m_semantics)
        
        # Normalize to get probability distribution across individual emotions
        betp = self.__pignistic_transform(m_fused)
        return betp

    def print_mass_function(self, mf, name):
        print(f"Evidence: {name}")
        for subset, mass in sorted(mf.items(), key=lambda x: -x[1]):
            if isinstance(subset, str):
                label = subset
            elif len(subset) == 1:
                label = next(iter(subset))
            else:
                label = "{" + ", ".join(sorted(subset)) + "}"

            scale = 30
            bar_length = int(mass * scale)
            bar = '=' * bar_length
            print(f"{label: <10}: {mass:.3f} | {bar}")

    def __incongruent(self, tone, face, semantics):
        tone_tense = self.__EMOTION_LIB.get(tone.emotion, self.__NEUTRAL_EMOTION)
        face_tense = self.__EMOTION_LIB.get(face.emotion, self.__NEUTRAL_EMOTION)
        semantics_tense = self.__EMOTION_LIB.get(semantics.emotion, self.__NEUTRAL_EMOTION)
        # print(f"Tone {tone_tense} Face {face_tense} Semantics {semantics_tense}")

        if tone_tense + face_tense > 0 and semantics_tense == self.__NEGATIVE_EMOTION:
            return True
        if tone_tense + face_tense < 0 and semantics_tense == self.__POSITIVE_EMOTION:
            return True
        
        return False

    def __discount_mass_function(self, m, alpha):
        discounted = {}
        
        for subset, mass in m.items():
            if subset == frozenset(self.__E):
                discounted[subset] = alpha * mass + (1 - alpha)
            else:
                discounted[subset] = alpha * mass
        
        return discounted

    def __combine_mass_functions(self, m1, m2):
        combined = {}
        conflict = 0.0

        # Iterate over all combinations of subset intersections
        for subset1, mass1 in m1.items():
            for subset2, mass2 in m2.items():
                intersection = subset1 & subset2
                product = mass1 * mass2

                if not intersection:
                    conflict += product
                else:
                    combined[intersection] = combined.get(intersection, 0.0) + product

        # Fallback
        if conflict == 1:
            return {self.__E: 1.0}
        
        # Normalize mass
        for subset in combined:
            combined[subset] /= (1 - conflict)
        
        return combined

    def __pignistic_transform(self, m):
        betp = {e: 0.0 for e in self.__E}
        for subset, mass in m.items():
            size = len(subset)
            if size == 0:
                continue
            for e in subset:
                betp[e] += mass / size
        return betp