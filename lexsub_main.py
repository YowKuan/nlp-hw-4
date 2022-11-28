#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow
import string

import gensim
import transformers 
import collections

from typing import List

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    all_lemmas = wn.lemmas(lemma, pos=pos)
    candidates = set()
    
    for l in all_lemmas:
        synset = l.synset()
        target_lemmas = synset.lemmas() 
        for target in target_lemmas:
            if target.name() != lemma: 
                candidates.add(target.name().replace('_', ' '))
    
    return list(candidates)

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    all_lemmas = wn.lemmas(context.lemma, pos=context.pos)
    counter = collections.Counter()
    
    for l in all_lemmas:
        synset = l.synset()
        target_lemmas = synset.lemmas() 
        for target in target_lemmas:
            if target.name() != context.lemma: 
                target_name = target.name().replace('_', ' ')
                counter[target_name] += target.count()
    return counter.most_common(1)[0][0]

def wn_simple_lesk_predictor(context : Context) -> str:
    stop_words = set(stopwords.words('english'))
    
    all_lemmas = wn.lemmas(context.lemma, pos=context.pos)
    counter = collections.Counter()
    
    #the Lemma objects are really lexemes = pairs of word and sense. 
    #The count method on that object returns how often that lexeme appeared in the semcor corpus. 
    max_intersection = 0
    final_synset = None
    for l in all_lemmas:
        synset = l.synset()
        definition_set = set(tokenize(synset.definition()))
        #You should therefore add the following to the definition:
        #All examples for the synset.
        examples = synset.examples()
        for example in examples:
            definition_set.update(set(tokenize(example)))
        #The definition and all examples for all hypernyms of the synset.
        hypernyms = synset.hypernyms()
        for hypernym in hypernyms:
            for example in hypernym.examples():
                definition_set.update(set(tokenize(example)))
            definition_set.update(set(tokenize(hypernym.definition())))
        def_set_wo_stopword = set()
        #remove stopwords
        for ele in definition_set:
            if ele not in stop_words:
                def_set_wo_stopword.add(ele)
                
        #compute intersection
        #To do: tokenize left/right context?
        print("left context", context.left_context)
        intersection_left = def_set_wo_stopword.intersection(set(context.left_context))
        intersection_right = def_set_wo_stopword.intersection(set(context.right_context))
        total_intersection = len(intersection_left) + len(intersection_right)
        print(total_intersection)
        
        #computing part b
        b = 0
        target_lemmas = synset.lemmas()
        for lemma in target_lemmas:
            if lemma.name() == context.lemma:
                b += lemma.count()
        
        if total_intersection > max_intersection:
            max_intersection = total_intersection
            final_synset = synset
    print("final_synset", final_synset)
    max_count = 0
    result = None
    final_lemmas = final_synset.lemmas()
    
    for lemma in final_lemmas:
        count = lemma.count()
        if count > max_count:
            max_count = count
            result = lemma
            
        
    return result #replace for part 3        
   

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        candidates = get_candidates(context.lemma, context.pos)
        max_similarity = 0
        result = None
        for candidate in candidates:
            try:
                similarity = self.model.similarity(context.lemma, candidate)
            except KeyError:
                continue
            if similarity > max_similarity:
                max_similarity = similarity
                result = candidate
        return result # replace for part 4


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        candidates = get_candidates(context.lemma, context.pos)
        sentence = ' '.join(context.left_context) + '[MASK]' + ' '.join(context.right_context)
        print(sentence)
        input_toks = self.tokenizer.encode(sentence)
        input_mat = np.array(input_toks).reshape((1,-1))
        outputs = self.model.predict(input_mat)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][5])[::-1]

        
        return self.tokenizer.convert_ids_to_tokens(best_words[0:1])[0] # replace for part 5

    

if __name__=="__main__":
    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)
    
    predictor_bert = BertPredictor()

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        #prediction = wn_frequency_predictor(context) 
        #prediction = predictor.predict_nearest(context)
        prediction = predictor_bert.predict(context)
        
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
