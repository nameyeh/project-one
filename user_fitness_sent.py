# Dependencies
import tweepy
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Twitter API Keys
from config import (consumer_key, 
                    consumer_secret, 
                    access_token, 
                    access_token_secret)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

def fit_sentiment(user):
    search_terms = ['fitness','exercise','strength','workout','goals','cardio','health','workouts','diet']
    compounds = []
    fitness_compounds = []
    for x in range(5):
        tweets = api.user_timeline(user, page=x+1)
        for tweet in tweets:
            text = tweet['text']
            results = analyzer.polarity_scores(text)
            if any(term in text for term in search_terms):
                fitness_compounds.append(results['compound'])
                compounds.append(None)
            else:
                compounds.append(results['compound'])
                fitness_compounds.append(None)

    x_axis = np.arange(len(compounds))
    fig,ax = plt.subplots()
    ax.scatter(x_axis,fitness_compounds,color='g',alpha=.8,edgecolor='black')
    ax.scatter(x_axis,compounds,color='r',alpha=.6,edgecolor='black')
    plt.title(f"Fitness Sentiment for User {user}")
    plt.ylabel("Tweet Polarity")
    plt.xlabel("Tweets Ago")
    plt.show()