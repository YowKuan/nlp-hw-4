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
            target_name = target.name().replace('_', ' ')
            if target_name != context.lemma: 
               
                counter[target_name] += target.count()
    return counter.most_common(1)[0][0]

def wn_simple_lesk_predictor(context : Context) -> str:
    stop_words = set(stopwords.words('english'))
    
    all_lemmas = wn.lemmas(context.lemma, pos=context.pos)
    
    # the Lemma objects are really lexemes = pairs of word and sense. 
    # The count method on that object returns how often that lexeme appeared in the semcor corpus. 
    final_score = collections.Counter()
    
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
        context_all = context.left_context + context.right_context
        processed_context =  tokenize(' '.join(context_all))
        total_intersection = len(def_set_wo_stopword.intersection(set(processed_context)))
        
        #computing part b
        b = 0
        target_lemmas = synset.lemmas()
        #print(target_lemmas)
        for lemma in target_lemmas:
            lemma_name = lemma.name().replace('_', ' ')
            if lemma_name == context.lemma:
                b += lemma.count()
            
        for lemma in target_lemmas:
            final_score[(synset, lemma.name())] =  total_intersection * 100000 + b * 1000 + lemma.count()

    for candidate in final_score.most_common():
        if candidate[0][1] != context.lemma:
            return candidate[0][1]     
   

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
        #also need to account for cls
        #mask_index = len(context.left_context) + 1
        sentence = ' '.join(context.left_context) + ' [MASK] ' + ' '.join(context.right_context)
        input_toks = self.tokenizer.encode(sentence)
        decoded = self.tokenizer.convert_ids_to_tokens(input_toks)
        mask_index = decoded.index('[MASK]')
        input_mat = np.array(input_toks).reshape((1,-1))
        outputs = self.model.predict(input_mat)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][mask_index])[::-1]
        output_tokens = self.tokenizer.convert_ids_to_tokens(best_words)
        for token in output_tokens:
            if token in set(candidates):
                return token

        
        return None # replace for part 5
    
class AllPredictor(object):
    
    def __init__(self, predictor, predictor_bert): 
        self.bert_predictor = predictor_bert
        self.word_to_vec_predictor = predictor

    def predict(self, context : Context) -> str:
    
        
        result = collections.Counter()
        for context in read_lexsub_xml(sys.argv[1]):
            #print(context)  # useful for debugging
            prediction1 = wn_simple_lesk_predictor(context) 
            result[prediction1]+=1
            prediction2 = wn_frequency_predictor(context)
            result[prediction2]+=1
            prediction3 = self.bert_predictor.predict(context)
            result[prediction3]+=1
            prediction4 = self.word_to_vec_predictor.predict_nearest(context)
            result[prediction4]+=1
        return result.most_common(1)[0][0]
    def predict2(self, context : Context) -> str:
        candidates = get_candidates(context.lemma, context.pos)
        
        result = None
        for candidate in candidates:
            try:
                similarity = self.model.similarity(context.lemma, candidate)
            except KeyError:
                continue


    

if __name__=="__main__":
    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)
    
    predictor_bert = BertPredictor()
    all_predictor = AllPredictor(predictor, predictor_bert)

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        #prediction = wn_simple_lesk_predictor(context) 
        #prediction = predictor.predict_nearest(context)
        #prediction = predictor_bert.predict(context)
        prediction = all_predictor.predict(context)
        
        
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
