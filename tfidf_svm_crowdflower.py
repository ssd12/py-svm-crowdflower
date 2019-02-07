import pandas as pd

#Download text_emotion.csv data from- www.figure-eight.com/data-for-everyone/ and place in same
#folder as this file
#Dataset: Emotion in Text dataset

tweets = pd.read_csv("./text_emotion.csv", usecols=["sentiment", "content"])

sentiment_counts = tweets.sentiment.value_counts()
sentiment_counts
#neutral       8638
#worry         8459
#happiness     5209
#sadness       5165
#love          3842
#surprise      2187
#fun           1776
#relief        1526
#hate          1323
#empty          827
#enthusiasm     759
#boredom        179
#anger          110

tweets.groupby('sentiment').describe()
 
import nltk
import string

def tokenize(message):
    """ removes punctuation and tokenizes the words .
    """
    msg = "".join([ch for ch in message if ch not in string.punctuation]) # get rid of punctuations
    tokens = nltk.word_tokenize(msg) 
    stems = [x.lower() for x in tokens] 
    return stems

tweets.content.head().apply(tokenize)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


fv = CountVectorizer(analyzer=tokenize).fit(tweets.content)
tweets_fv = fv.transform(tweets.content)
tfidf = TfidfTransformer().fit(tweets_fv)
tweets_tfidf = tfidf.transform(tweets_fv)

import json

fv_fix = {}
for k, v in fv.vocabulary_.items():
    fv_fix[k] = int(v)
    
with open('words_array.json', 'w') as fp:
    json.dump(fv_fix, fp)
    
idf = {}
idf['idf'] = tfidf.idf_.tolist()
with open('words_idf.json', 'w') as fp:
    json.dump(idf, fp)

from sklearn.svm import LinearSVC

model = LinearSVC().fit(tweets_tfidf, tweets.sentiment)
predictions = model.predict(tweets_tfidf)

from sklearn.metrics import accuracy_score, confusion_matrix
print('accuracy', accuracy_score(tweets['sentiment'], predictions))
print('confusion matrix\n', confusion_matrix(tweets['sentiment'], predictions))
print('(row=expected, col=predicted)')

import pickle
filename = 'tfidf_svm_crowdflower_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file, protocol = pickle.HIGHEST_PROTOCOL)

#model = pickle.load(open(filename, 'rb'))



import coremltools
coreml_model = coremltools.converters.sklearn.convert(model, "content", "sentiment")
coreml_model.short_description = "Classify emotion sentiment in a text"
coreml_model.input_description["content"] = "TFIDF of text"
coreml_model.output_description["sentiment"] = "Index of sentiment classes"
#save the model
coreml_model.save("tfidf_svm_crowdflower.mlmodel")
