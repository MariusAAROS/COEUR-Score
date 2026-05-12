import spacy
import numpy as np
from spacy.matcher import Matcher
from spellchecker import SpellChecker
from nltk.corpus import wordnet as wn

class USQA:
    def __init__(self, stories: list):
        self.stories = stories
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        self.h = SpellChecker()
        
        user_pattern = [
            [{"POS":"ADP"},{"POS":"DET"},{"POS":"NOUN"}],
            [{"POS":"ADP"},{"POS":"DET"},{"POS":"NOUN"},{"POS":"NOUN"}]
        ]
        something_pattern = [
            [{"POS":"PRON"},{"POS":"VERB"},{"POS":"PRON"}],
            [{"POS":"PRON"},{"POS":"VERB"},{"POS":"DET"},{"POS":"NOUN"}],
            [{"POS":"PRON"},{"POS":"VERB"},{"POS":"DET"},{"POS":"NOUN"},{"POS":"NOUN"}],
            [{"POS":"PRON"},{"POS":"VERB"},{"POS":"DET"},{"POS":"PROPN"},{"POS":"PROPN"},{"POS":"NOUN"}],
            [{"POS":"PRON"},{"POS":"VERB"},{"POS":"DET"},{"POS":"ADJ"},{"POS":"NOUN"}],
            [{"POS":"PRON"},{"POS":"AUX"},{"POS":"VERB"},{"POS":"PART"}]
        ]
        benefit_pattern = [
            [{"POS":"PART"},{"POS":"VERB"}],
            [{"POS":"PART"},{"POS":"VERB"},{"POS":"NOUN"}],
            [{"POS":"PART"},{"POS":"VERB"},{"POS":"NOUN"},{"POS":"PART","OP":"?"},{"POS":"NOUN"}],
            [{"POS":"PART"},{"POS":"VERB"},{"POS":"PRON"}],
            [{"POS":"PART"},{"POS":"VERB"},{"POS":"NOUN"},{"POS":"PRON"},{"POS":"NOUN"},{"POS":"PART", "OP":"?"},{"POS":"NOUN"}],
            [{"POS":"ADP"},{"POS":"NOUN"},{"POS":"NOUN"}],
            [{"POS":"SCONJ"},{"POS":"SCONJ"},{"POS":"PRON"},{"POS":"AUX"},{"POS":"VERB"}]
        ]
        self.matcher.add("USER_SOMETHING_BENEFIT", [*user_pattern,*something_pattern,*benefit_pattern])

    def compute_single_story(self, text):
        doc = self.nlp(text)
        matches = self.matcher(doc)
        spans = [doc[start:end] for _, start, end in matches]
        no_dupes = []
        suggestions = []
        words_pos = []
        net_synonym_set = []
        for span in spacy.util.filter_spans(spans):
            no_dupes.append(span.text)
            for i in span:
                if i.text != "'s":
                    if i.pos_ == "NOUN":
                        words_pos.append([i.text,"n"])
                    elif i.pos_ == "VERB":
                        words_pos.append([i.text,"v"])
                    else:
                        suggestions.append(self.h.candidates(i.text))
        for coll in words_pos:
            net_synonym_set.append(len(wn.synsets(coll[0],coll[1])))
        
        def Avg_Polysemies(words_synsets):
            if len(words_synsets) == 0:
                return 0
            return sum(words_synsets)/len(words_synsets)
            
        avg_poly_in_text = Avg_Polysemies(net_synonym_set)

        if len(no_dupes) >= 3:
            us_completeness_re = 1
        else:
            us_completeness_re = 0

        if len(doc) >= 13 and len(doc) < 18:
            us_usefulness_re = 1
        else:
            us_usefulness_re = 0
        
        us_polysemies = 1
        for synonym_set in net_synonym_set:
            if synonym_set == 0:
                us_polysemies = 0
            elif avg_poly_in_text >= 6:
                us_polysemies = 0
            else:
                us_polysemies = 1
        
        usqa_score = np.sum([us_completeness_re, us_polysemies, us_usefulness_re])

        return usqa_score
    
    def compute(self):
        usqa_scores = []
        for story in self.stories:
            usqa_scores.append(self.compute_single_story(story))
        return np.mean(usqa_scores) / 3