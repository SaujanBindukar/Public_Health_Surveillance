consumer_key = "fk8byb9V8aKdRHepMnulf6tI7";
consumer_secret = "ChQ1rHJb0DDmUfiEL0Rk1rjQsllBtLLLAbO3I9aEbS4vG2pN9z";
access_token = "918125352546738176-q1aVXcesbeHecmZcZryDZJGniHis4a1";
access_token_secret = "sXJmY61vVlzRY6D8z9D1MwkqUVmrEU8NBbQ9oDwByu4E7";

import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import csv

# Create the authentication object
authenticate = tweepy.OAuthHandler(consumer_key, consumer_secret)

# Set the access token and access token secret
authenticate.set_access_token(access_token, access_token_secret)

# Creating the API object while passing in auth information
api = tweepy.API(authenticate, wait_on_rate_limit=True, wait_on_rate_limit_notify=False)

# Create a dataframe with a column called Tweets
df = pd.DataFrame([tweet.text for tweet in tweepy.Cursor(api.search,q="coronavirus OR covid19 OR covidpandemic",count=100,lang="en",since="2020-01-26").items(100)], columns=['Tweets'])
df["Original_Text"]=[tweet.text for tweet in tweepy.Cursor(api.search,q="coronavirus OR covid19 OR covidpandemic",count=100,lang="en",since="2020-01-26").items(100)]
df["Locations"] = [tweet.user.location for tweet in tweepy.Cursor(api.search,q="coronavirus OR covid19 OR covidpandemic",count=100,lang="en",since="2020-01-26").items(100)]
df.to_csv(r'data.csv', index=False)

# Create a function to clean the tweets
def cleanTxt(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\r', ' ', text)
    text = re.sub('@[A-Za-z0–9]+', '', text)  # Removing @mentions
    text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', '', text)  # removing numbers
    text = re.sub('#', '', text)  # Removing '#' hash tag
    text = re.sub('RT[\s]+', '', text)  # Removing RT
    text = re.sub('https?:\/\/\S+', '', text)  # Removing hyperlink
    text = re.sub(r'R\$', ' ', text)  # removing special characters
    text = re.sub(r'\W', ' ', text)  # removing special characters
    text = re.sub(r'\s+', ' ', text)  # removing whitespace

    return text


# Clean the tweets
df['Tweets'] = df['Tweets'].apply(cleanTxt)
df.to_csv(r'data.csv', index=True)

# Show the cleaned tweets
print(df)


# Create a function to get the subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


# Create a function to get the polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity


def getSentiment(text):
    return TextBlob(text).sentiment


# Create two new columns 'Subjectivity' & 'Polarity'
df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
df.to_csv(r'data.csv', index=True)
df['Polarity'] = df['Tweets'].apply(getPolarity)
df.to_csv(r'data.csv', index=True)

df['Sentiment_Score'] = df['Tweets'].apply(getSentiment)
df.to_csv(r'data.csv', index=True)

# Show the new dataframe with columns 'Subjectivity' & 'Polarity'
print(df)

# word cloud visualization
allWords = ' '.join([twts for twts in df['Tweets']])
wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(allWords)

plt.imshow(wordCloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# Create a function to compute negative (-1), neutral (0) and positive (+1) analysis
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


df['Analysis'] = df['Polarity'].apply(getAnalysis)
df.to_csv(r'data.csv', index=True)


def getAnalysisText(score):
    if score < 0:
        return '0'
    elif score == 0:
        return '0'
    else:
        return '1'


df['sentiment'] = df['Polarity'].apply(getAnalysisText)
df.to_csv(r'data.csv', index=True)

# Show the dataframe
print(df)

# Printing positive tweets
print('Printing positive tweets:\n')
j = 1
sortedDF = df.sort_values(by=['Polarity'])  # Sort the tweets
for i in range(0, sortedDF.shape[0]):
    if (sortedDF['Analysis'][i] == 'Positive'):
        print(str(j) + ') ' + sortedDF['Tweets'][i])
        print()
        j = j + 1

# Printing negative tweets
print('Printing negative tweets:\n')
j = 1
sortedDF = df.sort_values(by=['Polarity'], ascending=False)  # Sort the tweets
for i in range(0, sortedDF.shape[0]):
    if (sortedDF['Analysis'][i] == 'Negative'):
        print(str(j) + ') ' + sortedDF['Tweets'][i])
        print()
        j = j + 1

# Plotting
plt1.figure(figsize=(8, 6))
for i in range(0, df.shape[0]):
    plt1.scatter(df["Polarity"][i], df["Subjectivity"][i], color='Blue')
# plt.scatter(x,y,color)
plt1.title('Sentiment Analysis')
plt1.xlabel('Polarity')
plt1.ylabel('Subjectivity')
plt1.show()

ptweets = df[df.Analysis == 'Positive']
ptweets = ptweets['Tweets']
print(ptweets)

print(round((ptweets.shape[0] / df.shape[0]) * 100, 1))

# Print the percentage of negative tweets
ntweets = df[df.Analysis == 'Negative']
ntweets = ntweets['Tweets']
print(ntweets)

print(round((ntweets.shape[0] / df.shape[0]) * 100, 1))

# Show the value counts
df['Analysis'].value_counts()

# Plotting and visualizing the counts
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
df['Analysis'].value_counts().plot(kind='bar')
plt.show()
