'''
Realtime classification code for classifying the tweets

Author : Vinod Verma

DateTime : 25/01/2021 
'''


import os
import re
import sys
import time
import pickle
import string
import tweepy 
import numpy as np
import pandas as pd
from datetime import datetime


import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Fill the X's with the credentials obtained from twitter API 
consumer_key = "xxxxxxxxxxxxxxxxxxxxx" 
consumer_secret = "xxxxxxxxxxxxxxxxxxxxx"
access_key = "xxxxxxxxxxxxxxxxxxxxx"
access_secret = "xxxxxxxxxxxxxxxxxxxxx"
  
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key,access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify = True,retry_count = 5,retry_delay = 5)


#Message cleaning
def message_cleaning(message):
    message = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",message).split())
    
    message = message.lower()
    
    stop_words = stopwords.words('english')+list(string.punctuation)

    Test_punc_removed_join_clean = [word for word in message.split() if word not in stop_words]
    
    wordnet_lemmatizer = WordNetLemmatizer()
    
    lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in Test_punc_removed_join_clean]
    
    return lemma_list



def trainModelPreprocess(masterData):

    # Spell Correction Function
    def SpellCorrect(raw_data):
		
        # To find all matching regex Values
        raw_data = raw_data.apply(lambda x: " ".join(re.findall(r"[a-zA-zA-Z0-9/-]+", x)))
        
        # Convert the data to lower Case
        raw_data = raw_data.str.lower()
        
        return raw_data
	
    # Call Spell Check Function
    masterData['replyText'] = SpellCorrect(masterData['replyText'])
    	
    print("[INFO] : Spell Correction Done")
    
    # Work Tokenize
    masterData['replyText'] = masterData['replyText'].apply(word_tokenize)

    stop_words = stopwords.words('english')+list(string.punctuation)
    
    def remove_stopWord(raw_data):
        raw_data = raw_data.apply(lambda x:[w for w in x if  w not in (stop_words)])
        return raw_data
        
    masterData['replyText'] = remove_stopWord(masterData['replyText'])
    
    print("[INFO] : StopWord Removal Done")

    # For Word Lemmatizing Operation
    lm=WordNetLemmatizer()
    
    def convert_lemma(raw_data, lm):
        raw_data = raw_data.apply(lambda x: [lm.lemmatize(word) for word in x])
        return raw_data
        
    masterData['replyText'] = convert_lemma(masterData['replyText'], lm)
    
    print("[INFO] : Word Lemmatization Done")

    # Converting data into DataFrame
    masterData['replyText'] = pd.DataFrame(masterData['replyText'])
    
    # Apply String
    masterData['replyText'] = masterData['replyText'].apply(str)
    
    # Transforming Data into Desired Format for Further Processing
    FeatureDataMatrix = []
    for each in masterData['replyText']:
        FeatureDataMatrix.append(each)

    # # Saving Model using Pickle
    tfidfVectPath = open('savedata/tfidf_vector.pickle', 'rb')
    tfidf_vectorizer_Model = pickle.load(tfidfVectPath)
    tfidfVectPath.close()
       
    # # Saving Model using Pickle
    NLP_modelPath = open('savedata/NLP_model_Jb.pickle', 'rb')
    multiNBmodel = pickle.load(NLP_modelPath)
    NLP_modelPath.close()

    # Vectorizing the Data
    input_vector = tfidf_vectorizer_Model.transform(FeatureDataMatrix)
    
    # Predicting from the available model
    result = multiNBmodel.predict(input_vector)
    
    # Adding Prediction result to dataFrame
    masterData['bullPred'] = result
    
    print("[INFO] : Prediction Done")
    
    # Saving the Results File
    masterData.to_csv('results.csv', header = True)
    
    return masterData


#Entering the search keyword and classifying tweets belonging to the search keyword
print("\nEnter the search keyword: ")
query = input()

startTime = datetime.now()

c=0
tweet_count = 0
bully_count = 0
numofItems = 10
numofreply = 50
replyList = []
bully_record = []
date_since = '2021-01-11'


tweets = tweepy.Cursor(api.search, q=query,lang="en", since=date_since,result_type='popular', timeout=999999).items(numofItems)

for tweet in tweets:
    cleaned_tweet = tweet.text
    tweet_id = tweet.id
    poster = tweet.user.screen_name
    
    count = 0
    reTweets = tweepy.Cursor(api.search,q="@"+poster, since_id=tweet_id).items(numofreply)
    
    for reply in reTweets:
    
        if(reply.in_reply_to_status_id == tweet_id):
            
            tweet_count += 1
            
            inputTextarray = [poster, str(tweet_id), cleaned_tweet,reply.user.screen_name, poster, reply.text]
            
            print(inputTextarray)
            
            replyList.append(inputTextarray)
            
            count = count + 1
            
        if count==numofItems:
            break    
            
        c+=1
        if c % 1000 == 0:  # first request completed, sleep 5 sec
            print("\nSleeping for 10 sec\n")
            time.sleep(10)
            

# Header List for DataFrame
headers = ['Poster', 'TweetID', 'Tweet', 'replyUserName', 'Poster_1', 'replyText'] 

# Converting output to DataFrame
tweetDF = pd.DataFrame(replyList, columns = headers)
        
# Preprocessing the input data        
tweetDFpred = trainModelPreprocess(tweetDF)

# Check the Output if reply to a tweet is bullying or not
bullyingDF = tweetDFpred.loc[tweetDFpred['bullPred'] == 1]

# Total Tweet Count
tweet_count = int(tweetDFpred.shape[0])

# Total bullying tweets 
bully_count = int(bullyingDF.shape[0])

print("====================================================")
print("===============Final Observations===================")
print("====================================================")
print("[INFO] - Total tweet     :" + str(tweet_count))
print("[INFO] - Offensive tweets:" + str(bully_count))


#Offendor,victim and tweet in each record
offenders = []
victims = []

# check if bullying tweet is present or not
if bully_count > 0:
    print("\nFollowing tweets were detected as bullying : ")
    # index=0
    
    # loop for getting offenders and Victims 
    # for indx in range(bullyingDF.shape[0]):
    for indx in bullyingDF.index:
        # index+=1
        offenders.append(bullyingDF['Poster_1'][indx])
        victims.append(bullyingDF['replyUserName'][indx])
            
    # Final List of Offenders and Victims
    offenders = list(dict.fromkeys(offenders))
    victims = list(dict.fromkeys(victims))  
    
        

# Condition to check offenders     
if len(offenders) > 0:
    for eachInd in range(len(offenders)):
        print('  [INFO] - Mr./Mrs./Ms. {} Offended the Mr./Mrs./Ms. {}'.format(offenders[eachInd], victims[eachInd]))

print("====================================================")
print("====================================================")

endTime = datetime.now()
executionTime = endTime - startTime
print('Execution Time : ', executionTime)
