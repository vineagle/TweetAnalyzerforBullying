'''
Training Code for classifying the tweets into bullying or not.

Author : Vinod Verma

DateTime : 25/01/2021

Model Precision : 0.93
Model Recall    : 0.74

'''

import re
import pickle
import string
import numpy as np
import pandas as pd
from datetime import datetime

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer

startTime = datetime.now()

masterData = pd.read_csv('dataset.csv', header = 0, encoding='latin1')

# 'tweet' , 'bullying'

masterData = masterData.dropna()

def trainModel(masterData):

    # Spell Correction Function
    def SpellCorrect(raw_data):
		
        # To find all matching regex Values
        raw_data = raw_data.apply(lambda x: " ".join(re.findall(r"[a-zA-zA-Z0-9/-]+", x)))
        
        raw_data = raw_data.str.lower()
        
        return raw_data
	
    # Call Spell Check Function
    masterData['tweet'] = SpellCorrect(masterData['tweet'])
    
    print("[INFO] : Spell Correction Done")
    
    masterData['tweet'] = masterData['tweet'].apply(word_tokenize)

    stop_words = stopwords.words('english')+list(string.punctuation)
    
    def remove_stopWord(raw_data):
        raw_data = raw_data.apply(lambda x:[w for w in x if  w not in (stop_words)])
        return raw_data
        
    masterData['tweet'] = remove_stopWord(masterData['tweet'])
    
    print("[INFO] : StopWord Removal Done")

    # For Word Lemmatizing Operation
    lm=WordNetLemmatizer()
    
    def convert_lemma(raw_data, lm):
        raw_data = raw_data.apply(lambda x: [lm.lemmatize(word) for word in x])
        return raw_data
        
    masterData['tweet'] = convert_lemma(masterData['tweet'], lm)
    
    print("[INFO] : Word Lemmatization Done")

    masterData['tweet'] = pd.DataFrame(masterData['tweet'])
    
    masterData['tweet'] = masterData['tweet'].apply(str)
    
    # Converting Data into Desired Format for Further Processing
    FeatureDataMatrix = []
    for each in masterData['tweet']:
        FeatureDataMatrix.append(each)
    
    
    # settings that you use for count vectorizer will go here 
    # tfidf_vectorizer=TfidfVectorizer(use_idf=True, lowercase=False, ngram_range=(1,2), min_df=4, max_features=100) 
    tfidf_vectorizer=TfidfVectorizer(use_idf=True, lowercase=False, ngram_range=(1,2), min_df=15, max_features=1000) 
    
    # just send in all your docs here 
    tfidf_vectorizer_Model=tfidf_vectorizer.fit(FeatureDataMatrix)
    
    X_train_tfidf = tfidf_vectorizer_Model.transform(FeatureDataMatrix)
    
    # Saving Model using Pickle
    tfidfVectPath = open('savedata/tfidf_vector.pickle', 'wb')
    pickle.dump(tfidf_vectorizer_Model, tfidfVectPath)
    tfidfVectPath.close()
    
    # MultinomialNB Model Only for (Positive Feature Values)
    multiNBmodel = MultinomialNB(alpha=0.1)
    multiNBmodel = multiNBmodel.fit(X_train_tfidf, np.array(masterData['bullying']))	

    NLP_modelPath = open('savedata/NLP_model_Jb.pickle', 'wb')
    pickle.dump(multiNBmodel, NLP_modelPath)
    NLP_modelPath.close()	
    
    print("[INFO] : ensemble Model Saved")
    
    # Test Data Prediction
    predicted_Val = multiNBmodel.predict(X_train_tfidf)
    
    masterData['pred'] = predicted_Val
    
    print("[INFO] : Prediction Done")
    
    masterData.to_csv('results.csv', header = True)
    
    
    cfMatrix = confusion_matrix(masterData['bullying'], masterData['pred'])
    precisionScore = precision_score(masterData['bullying'], masterData['pred'])
    recallScore = recall_score(masterData['bullying'], masterData['pred']) 

    print(cfMatrix)
    print(precisionScore)
    print(recallScore)
    
    return {'status': True, 'precisionScore': precisionScore, 'recallScore' : recallScore}


res = trainModel(masterData)

endTime = datetime.now()
executionTime = endTime - startTime
print("==========================================================")
print('Execution Time : ', executionTime)
print("==========================================================")