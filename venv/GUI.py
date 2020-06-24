from tkinter import *

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from PIL import Image, ImageTk
import requests
import json
import csv
import pandas as pd
import tkinter.ttk as ttk
import tweepy
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
from textblob import TextBlob
from wordcloud import WordCloud
from langdetect import detect
from tkcalendar import  *
import datetime
import dateutil.parser
import numpy as np
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import learning_curve
from sklearn.base import BaseEstimator, TransformerMixin
from tkinter import messagebox

import tkinter as tk
from tkinter.filedialog import askopenfilename
import pycountry

consumer_key = "fk8byb9V8aKdRHepMnulf6tI7";
consumer_secret = "ChQ1rHJb0DDmUfiEL0Rk1rjQsllBtLLLAbO3I9aEbS4vG2pN9z";
access_token = "918125352546738176-q1aVXcesbeHecmZcZryDZJGniHis4a1";
access_token_secret = "sXJmY61vVlzRY6D8z9D1MwkqUVmrEU8NBbQ9oDwByu4E7";





class Fetch():
    def __init__(self):
        self.fetch=Tk()
        self.fetch.geometry("1020x680+160+10")
        self.fetch.title("Opinion Mining System")
        self.fetch.resizable(0, 0)
        self.fetchform()
        self.fetch.mainloop()
    def fetchform(self):
        Label(self.fetch, text="Fetch the Tweets", font=("w 30 bold")).place(x=350, y=15)
        entry1 = Entry(self.fetch)
        entry1.place(x=100, y=70, height=50, width=800)

        def exit():
            process.kill()


        def printing():



            # hash_tag_list = ["corona virus", "covid19", "covid-19", "coronapandemic", "coronavirus"]
            class TwitterStreamer():
                def __init__(self):
                    pass

                def stream_tweets(self, fetched_tweets_filename, hash_tag_list):
                    listener = StdOutListener(fetched_tweets_filename)
                    auth = OAuthHandler(consumer_key, consumer_secret)
                    auth.set_access_token(access_token, access_token_secret)
                    stream = Stream(auth, listener)
                    stream.filter(track=hash_tag_list)

            class StdOutListener(StreamListener):
                def __init__(self, fetched_tweets_filename):  # constructor
                    self.fetched_tweets_filename = fetched_tweets_filename

                def on_data(self, data):
                    try:
                        with open(self.fetched_tweets_filename, 'a') as tf:
                            tf.write(data)
                            print(data)

                        return True
                    except BaseException as e:
                        print("Error on_data %s" % str(e))
                    return True

                def on_error(self, status):
                    print(status)

            hash_tag_list = entry1.get()
            fetched_tweets_filename = "streamTweets.json"
            twitter_streamer = TwitterStreamer()
            twitter_streamer.stream_tweets(fetched_tweets_filename, hash_tag_list)
            Label(self.fetch, text="Fetch the Tweets", font=("w 30 bold")).place(x=350, y=500)

        Button(self.fetch, text="Start Fetching", command=printing).\
            place(x=300, y=150, height=50,width=100)
        Button(self.fetch, text="Stop Fetching", command=exit). \
            place(x=600, y=150, height=50, width=100)



class FirstPage():
    def __init__(self):
        self.root = Tk()
        self.root.geometry("1020x680+160+10")
        self.root.title("Opinion Mining System")
        self.root.resizable(0, 0)
        frame1 = Frame(self.root, height=680, width=510, bg="grey")
        frame1.pack(side=LEFT, fill=BOTH)
        bg_img = PhotoImage(file="assets/phs.png")
        bg_label = Label(frame1, image=bg_img)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.homePage()
        self.root.mainloop()



    def homePage(self):

        frame2 = Frame(self.root, height=680, width=510)
        frame2.pack(fill=BOTH),

        text1 = Label(frame2, text="OPINION MINING SYSTEM \n"
                                   "FOR PUBLIC HEALTH SURVEILLANCE \n"
                                   "USING TWEETS BASED ON LOCATION", font=" MonotypeCorsive 25 bold",pady=20 )
        text1.pack()

        text2 = Label(frame2, text="Get to know the opinion of the people from the tweets "
                                   "\n regarding the health issue", font="m 20",pady=10)
        text2.pack()

        frame4=Frame(self.root,)
        frame4.place(x=580, y=250)
        load = Image.open("assets/graph.jpg")
        render = ImageTk.PhotoImage(load)
        img = Label(frame4, image=render)
        img.image = render
        img.pack()

        frame3 = Frame(self.root, height=100)
        frame3.pack(fill=BOTH, anchor=SE, side=BOTTOM)

        btn1 = Button(frame3, text="Get Started", command=FetchTweets, padx=40, pady=10)
        btn1.pack(side=RIGHT, anchor=S, padx=40, pady=20)

        btn2 = Button(frame3, text="Get Corona Data", command=CoronaData, padx=40, pady=10)
        btn2.pack(side=LEFT, anchor=S, padx=40, pady=20)

class CoronaData():
    def __init__(self):
        self.coronawindow=Tk()
        self.coronawindow.geometry("1020x680+160+10")
        self.coronawindow.title("Corona Virus Tracker App")
        self.coronawindow.resizable(0, 0)
        self.showWorldData()
        self.showCountryData()
        self.coronawindow.mainloop()

    def showWorldData(self):
        response = requests.get("https://corona.lmao.ninja/v2/all")
        data=response.json()

        Label(self.coronawindow, text="Corona Virus Tracker" , font=("m 30")).place(x=350, y=10)
        f1=Frame(self.coronawindow, background="#2ecc71",)
        f1.place(x=25, y=70)
        Label(f1, text="Total Cases", background="#2ecc71", foreground="white").pack()
        Label(f1, text=data['cases'], background="#2ecc71", foreground="white").pack()

        f2 = Frame(self.coronawindow, bg="violet")
        f2.place(x=175, y=70)
        Label(f2, text="Today Cases", background="violet", foreground="white").pack()
        Label(f2, text=data['todayCases'], background="violet", foreground="white").pack()

        f3 = Frame(self.coronawindow, bg="#8e44ad",)
        f3.place(x=325, y=70)
        Label(f3, text="Total Deaths", background="#8e44ad", foreground="white").pack()
        Label(f3, text=data['deaths'], background="#8e44ad", foreground="white").pack()

        f4 = Frame(self.coronawindow, bg="red",)
        f4.place(x=475, y=70)
        Label(f4, text="Today Deaths", background="red", foreground="white").pack()
        Label(f4, text=data['todayDeaths'], background="red", foreground="white").pack()

        f5 = Frame(self.coronawindow, bg="#e74c3c",)
        f5.place(x=625, y=70)
        Label(f5, text=" Recovered ", background="#e74c3c", foreground="white").pack()
        Label(f5, text=data['recovered'], background="#e74c3c", foreground="white").pack()

        f5 = Frame(self.coronawindow, bg="green",)
        f5.place(x=775, y=70)
        Label(f5, text=" Total Active", background="green", foreground="white").pack()
        Label(f5, text=data['active'], background="green", foreground="white").pack()

        f6 = Frame(self.coronawindow, bg="orange",)
        f6.place(x=925, y=70)
        Label(f6, text="    Critical    ", background="orange", foreground="white").pack()
        Label(f6, text=data['critical'], background="orange", foreground="white").pack()


    def showCountryData(self):
        response = requests.get("https://corona.lmao.ninja/v2/countries")
        data = response.json()

        Label(self.coronawindow, text="Country Data", font=("m 30")).place(x=370, y=130)
        search1= Entry(self.coronawindow, font=("m 20")).place(x=0, y=170)

        def searchData(a):
            print(a)

        # Button(self.coronawindow, text="Search", font=("m 20"), command=searchData(search1.get())).place(x=370, y=170)

        tree = ttk.Treeview(self.coronawindow)
        # number of columns
        tree["columns"] = ("1", "2", "3", "4", "5", "6", "7", "8")
        tree.column("#0", width=50, stretch=YES)
        tree.column("1", width=170, stretch=YES)
        tree.column("2", width=120, stretch=YES)
        tree.column("3", width=120, stretch=YES)
        tree.column("4", width=120, stretch=YES)
        tree.column("5", width=120, stretch=YES)
        tree.column("6", width=120, stretch=YES)
        tree.column("7", width=100, stretch=YES)
        tree.column("8", width=120, stretch=YES)
        # heading of the table
        tree.heading("#0", text="S.N.", )
        tree.heading("1", text="Country")
        tree.heading("2", text="Cases", )
        tree.heading("3", text="Today Cases", )
        tree.heading("4", text="Deaths", )
        tree.heading("5", text="Today Deaths", )
        tree.heading("6", text="Recovered", )
        tree.heading("7", text="Active", )
        tree.heading("8", text="Critical", )

        tree.place(x=0, y=210, width=1020, height=500)
        i = 0
        for s in data:
            i = i + 1;
            tree.insert("", i, text=i, values=(s['country'], s['cases'],s['todayCases'],s['deaths'],s['todayDeaths'],s['recovered'],s['active'],s['critical']))

class FetchTweets():

    def __init__(self):
        self.root2 = Tk()
        self.root2.geometry("1020x680+160+10")
        self.root2.title("Fetching the tweets")
        self.root2.resizable(0, 0)
        self.fetchForm()
        self.root2.mainloop()



    def fetchForm(self):
        Label(self.root2, text="Fetch the Tweets", font=("w 30 bold")).place(x=350, y=15)

        formFrame=Frame(self.root2, height=500, width=700,borderwidth=6,background="lightblue" , relief=RAISED,)
        formFrame.pack(pady=80)

        keyword=Label(formFrame, text="Enter the keyword", font=" m 20", background="lightblue" )
        keyword.place(x=40, y=30)
        keywordEntry=Entry(formFrame,)
        keywordEntry.place(x=300, y=25,width=300,height=40)


        number = Label(formFrame, text="Enter the number ", font=" m 20", background="lightblue")
        number.place(x=40, y=105)
        numberEntry = Entry(formFrame, )
        numberEntry.place(x=300, y=100, width=300, height=40)

        def open_cal1():
            top=Toplevel(formFrame)
            top.geometry("250x180+700+310")


            print("opening calender")
            cal=Calendar(top,)
            cal.pack()
            def get_date():
                st_date = cal.get_date()
                datetimeobject = datetime.datetime.strptime(st_date, '%m/%d/%y').strftime('%Y-%m-%d')
                print("Starting date", datetimeobject)
                startEntry.delete(0, "end")
                startEntry.insert(0, datetimeobject)
                top.destroy()
            Button(top,text="Select Date",command=get_date).pack()


        start_date = Label(formFrame, text="Enter the start date ", font=" m 20", background="lightblue" )

        start_date.place(x=40, y=180)
        startEntry = Entry(formFrame)
        Button(formFrame,text="C", command=open_cal1).place(x=600, y=175,height=40)

        startEntry.place(x=300, y=175, width=300, height=40)


        def open_cal2():
            top = Toplevel(formFrame)
            top.geometry("250x180+700+390")
            print("opening calender")
            cal = Calendar(top, )
            cal.pack()

            def get_date():
                end_date = cal.get_date()
                datetimeobject=datetime.datetime.strptime(end_date, '%m/%d/%y').strftime('%Y-%m-%d')
                print("Starting date", datetimeobject)
                endEntry.delete(0, "end")
                endEntry.insert(0, datetimeobject)
                top.destroy()


            Button(top, text="Select Date", command=get_date).pack()


        end_date = Label(formFrame, text="Enter the end date ", font=" m 20", background="lightblue")
        end_date.place(x=40, y=265)
        endEntry = Entry(formFrame, )
        Button(formFrame, text="C", command=open_cal2).place(x=600, y=260, height=40)
        endEntry.place(x=300, y=260, width=300, height=40)


        def fetchData():
            query=keywordEntry.get()
            number=int(numberEntry.get())
            d1=startEntry.get()
            d2=endEntry.get()
            print(query)
            print(number)
            print(d1)
            print(d2)

            # Create the authentication object
            authenticate = tweepy.OAuthHandler(consumer_key, consumer_secret)

            # Set the access token and access token secret
            authenticate.set_access_token(access_token, access_token_secret)

            # Creating the API object while passing in auth information
            api = tweepy.API(authenticate, wait_on_rate_limit=True, wait_on_rate_limit_notify=False)
            print(api)

            # Create a dataframe with a column called Tweets
            df = pd.DataFrame([tweet.text for tweet in
                               tweepy.Cursor(api.search, q=query, lang="en",
                                           since=d1, until=d2  ).items(number)], columns=['text'])
            df["created_at"] = [tweet.created_at for tweet in
                                tweepy.Cursor(api.search, q=query,lang="en",
                                              since=d1, until=d2 ).items(number)]
            df["Location"] = [tweet.user.location for tweet in
                               tweepy.Cursor(api.search, q=query,lang="en",
                                             since=d1, until=d2  ).items(number)]
            df.to_csv(r'new_data.csv', index=True)
            print(df)
            SentimentAnalysisFetch()

        def clear_all_fields():
            keywordEntry.delete(0, END)
            numberEntry.delete(0, END)
            startEntry.delete(0, END)
            endEntry.delete(0, END)

        def import_csv_data():
            global v
            global csv_file_path
            csv_file_path = askopenfilename()
            print(csv_file_path)
            SentimentAnalysisCSV()
            # v.set(csv_file_path)
            # df = pd.read_csv(csv_file_path)

        btn1=Button(formFrame, text="Fetch Tweets", font=" m 20", command=fetchData,fg="green",pady=10, padx=10,relief=RAISED,
                    background="orange"
                    )
        btn1.place(x=20, y=400)
        btn2 = Button(formFrame, text=" Clear All", font=" m 20", command=clear_all_fields, fg="red", pady=10, padx=10,
                      )
        btn2.place(x=230, y=400)

        btn2 = Button(formFrame, text="Quit", font=" m 20",command=FirstPage, fg="red", pady=10, padx=30,
                      )
        btn2.place(x=380, y=400)

        btn3 = Button(formFrame, text="Open CSV", font=" m 20", command=import_csv_data, fg="red", pady=10, padx=10,
                      )
        btn3.place(x=530, y=400)
class SentimentAnalysisFetch():
    def __init__(self):
        self.root3 = Tk()
        self.root3.geometry("1020x680+160+10")
        self.root3.title("Fetching the tweets")
        self.root3.resizable(0, 0)
        self.fetchForm()
        self.analyse()
        self.root3.mainloop()

    def fetchForm(self):
        Label(self.root3, text="Twitter Data", font=("m 30")).place(x=370, y=5)
        tree = ttk.Treeview(self.root3)
        # number of columns
        tree["columns"] = ("1", "2", "3")
        tree.column("#0", width=50, stretch=YES)
        tree.column("1", width=200, stretch=YES)
        tree.column("2", width=800, stretch=YES)
        # tree.column("3", width=150, stretch=YES)

        # heading of the table
        tree.heading("#0", text="S.N.", )
        tree.heading("1", text="Created At")
        tree.heading("2", text="Tweets", )
        # tree.heading("3", text="Location", )

        tree.place(x=0, y=50, width=1020,height=500)
        i=0


        with open('new_data.csv') as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                i=i+1
                tweets = row['text']
                created_at=row['created_at']
                # locations=row['Locations']
                tree.insert("", i, text=i,values=(created_at,tweets))


    def analyse(self):
        f1=Frame(self.root3, bg="red", height=50, width=200)
        f1.pack(side=BOTTOM, anchor=E,padx=20,pady=10)

        def cleanTweets():
            def cleanTxt(text):
                text = re.sub(r'\n', ' ', str(text))
                text = re.sub(r'\r', ' ', text)
                text = re.sub('@[A-Za-z0–9]+', '', text)  # Removing @mentions
                text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', '', text)  # removing numbers
                text = re.sub('#', '', text)  # Removing '#' hash tag
                text = re.sub('RT[\s]+', '', text)  # Removing RT
                text = re.sub('https?:\/\/\S+', '', text)  # Removing hyperlink
                text = re.sub(r'R\$', ' ', text)  # removing special characters
                text = re.sub(r'\W', ' ', text)  # removing special characters
                text = re.sub(r'\s+', ' ', text) # removing whitespace
                return text

            def detectLanguage(text):
                try:
                    if (detect(text) == "en"):
                        return text
                    else:
                        return "Other Language"
                except Exception as ex:
                    pass

            def changeDate(text):
                return dateutil.parser.parse(text).date()

            def findCountry(text):
                for country in pycountry.countries:
                    if country.name in text:
                        return country.name



            df=pd.read_csv('new_data.csv')
            df["Language"]=df["text"].apply(detectLanguage)
            df.to_csv("languagetweets.csv", index=False)
            print("Printing langauge tweets")
            print(df)

            print("Deleting the other language tweets")
            data = pd.read_csv("languagetweets.csv")

            # Create a DataFrame object
            empDfObj = pd.DataFrame(data, columns=['Language'])
            if 'Other Language' not in empDfObj.values:
                print("other language not found")
                data.to_csv("final.csv", index=False)
            else:
                print("other language  found")
                data = pd.read_csv("languagetweets.csv")
                data = data.set_index("Language")
                data = data.drop("Other Language", axis=0)
                data.to_csv("final.csv", index=False)
                print(data)

            # data = data.set_index("Language")
            # data = data.drop("Other Language", axis=0)
            # data.to_csv("final.csv", index=False)
            # print(data)


            df = pd.read_csv('final.csv')
            df["Clean Tweets"]=df["text"].apply(cleanTxt)
            df.to_csv(r'cleandata.csv', index=False)
            print("Before null value")
            print(df['Clean Tweets'].isnull().sum())

            print("Removing null value")
            data = pd.read_csv("cleandata.csv")
            data['Clean Tweets'].replace('', np.nan, inplace=True)
            data.dropna(subset=['Clean Tweets'], inplace=True)
            data.to_csv(r'nonemptydata.csv', index=False)
            print("After removing null value")

            d1 = pd.read_csv("nonemptydata.csv")
            d1["created_at"] = d1["created_at"].apply(changeDate)
            d1.to_csv("nonemptydata.csv", index=False)

            data = pd.read_csv('nonemptydata.csv')
            if "user" in data:
                df = []
                for i in range(0, len(data['user'])):
                    b = eval(data['user'][i])
                    df.append(b["location"])
                data["Location"] = df
                data.to_csv("nonemptydata.csv", index=False)
            else:
                data.to_csv("nonemptydata.csv", index=False)

            df = pd.read_csv('nonemptydata.csv')
            df["Location"] = data["Location"].astype(str)
            df["Location"] = df["Location"].apply(findCountry)
            df.to_csv(r'nonemptydata.csv', index=False)



            cleanTweetsPage()
        Button(f1, text="Clean Tweets", command=cleanTweets).place(x=0,y=0, height=50,width=200,)
class SentimentAnalysisCSV():
    def __init__(self):
        self.root3 = Tk()
        self.root3.geometry("1020x680+160+10")
        self.root3.title("Fetching the tweets")
        self.root3.resizable(0, 0)
        self.fetchForm()
        self.analyse()
        self.root3.mainloop()

    def fetchForm(self):
        Label(self.root3, text="Twitter Data", font=("m 30")).place(x=370, y=5)
        tree = ttk.Treeview(self.root3)
        # number of columns
        tree["columns"] = ("1", "2", "3")
        tree.column("#0", width=50, stretch=YES)
        tree.column("1", width=200, stretch=YES)
        tree.column("2", width=800, stretch=YES)
        # tree.column("3", width=150, stretch=YES)

        # heading of the table
        tree.heading("#0", text="S.N.", )
        tree.heading("1", text="Created At")
        tree.heading("2", text="Tweets", )
        # tree.heading("3", text="Location", )

        tree.place(x=0, y=50, width=1020,height=500)
        i=0


        with open(csv_file_path) as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                i=i+1
                tweets = row['text']
                created_at=row['created_at']
                # locations=row['Locations']
                tree.insert("", i, text=i,values=(created_at,tweets))


    def analyse(self):
        f1=Frame(self.root3, bg="red", height=50, width=200)
        f1.pack(side=BOTTOM, anchor=E,padx=20,pady=10)

        def cleanTweets():
            def cleanTxt(text):
                text = re.sub(r'\n', ' ', str(text))
                text = re.sub(r'\r', ' ', text)
                text = re.sub('@[A-Za-z0–9]+', '', text)  # Removing @mentions
                text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', '', text)  # removing numbers
                text = re.sub('#', '', text)  # Removing '#' hash tag
                text = re.sub('RT[\s]+', '', text)  # Removing RT
                text = re.sub('https?:\/\/\S+', '', text)  # Removing hyperlink
                text = re.sub(r'R\$', ' ', text)  # removing special characters
                text = re.sub(r'\W', ' ', text)  # removing special characters
                text = re.sub(r'\s+', ' ', text) # removing whitespace
                return text

            def detectLanguage(text):
                try:
                    if (detect(text) == "en"):
                        return text
                    else:
                        return "Other Language"
                except Exception as ex:
                    pass

            def changeDate(text):
                return dateutil.parser.parse(text).date()

            def findCountry(text):
                for country in pycountry.countries:
                    if country.name in text:
                        return country.name



            df=pd.read_csv(csv_file_path)
            df["Language"]=df["text"].apply(detectLanguage)
            df.to_csv("languagetweets.csv", index=False)
            print("Printing langauge tweets")
            print(df)


            print("Deleting the other language tweets")
            data = pd.read_csv("languagetweets.csv")

            # Create a DataFrame object
            empDfObj = pd.DataFrame(data, columns=['Language'])
            if 'Other Language' not in empDfObj.values:
                print("other language not found")
                data.to_csv("final.csv", index=False)
            else:
                print("other language  found")
                data = pd.read_csv("languagetweets.csv")
                data = data.set_index("Language")
                data = data.drop("Other Language", axis=0)
                data.to_csv("final.csv", index=False)
                print(data)


            df = pd.read_csv('final.csv')
            df["Clean Tweets"]=df["text"].apply(cleanTxt)
            df.to_csv(r'cleandata.csv', index=False)
            print("Before null value")
            print(df['Clean Tweets'].isnull().sum())

            print("Removing null value")
            data = pd.read_csv("cleandata.csv")
            data['Clean Tweets'].replace('', np.nan, inplace=True)
            data.dropna(subset=['Clean Tweets'], inplace=True)
            data.to_csv(r'nonemptydata.csv', index=False)
            print("After removing null value")

            d1 = pd.read_csv("nonemptydata.csv")
            d1["created_at"] = d1["created_at"].apply(changeDate)
            d1.to_csv("nonemptydata.csv", index=False)



            data = pd.read_csv('nonemptydata.csv')
            if "user" in data:
                df = []
                for i in range(0, len(data['user'])):
                    b = eval(data['user'][i])
                    df.append(b["location"])
                data["Location"] = df
                data.to_csv("nonemptydata.csv", index=False)
            else:
                data.to_csv("nonemptydata.csv", index=False)

            df = pd.read_csv('nonemptydata.csv')
            df["Location"] = data["Location"].astype(str)
            df["Location"] = df["Location"].apply(findCountry)
            df.to_csv(r'nonemptydata.csv', index=False)



            cleanTweetsPage()
        Button(f1, text="Clean Tweets", command=cleanTweets).place(x=0,y=0, height=50,width=200,)

class cleanTweetsPage():
    def __init__(self):
        self.cleanTweetsWindow=Tk()
        self.cleanTweetsWindow.geometry("1020x680+160+10")
        self.cleanTweetsWindow.title("Cleaned Text")
        self.cleanTweetsWindow.resizable(0, 0)
        self.showCleanText()
        self.calculateSentiment()
        self.cleanTweetsWindow.mainloop()

    def showCleanText(self):
        Label(self.cleanTweetsWindow, text="Cleaned Data", font=("m 30")).place(x=370, y=5)
        tree = ttk.Treeview(self.cleanTweetsWindow)

        # number of columns
        tree["columns"] = ("1", "2", "3","4")
        tree.column("#0", width=50, stretch=YES)
        tree.column("1", width=100, stretch=YES)
        tree.column("2", width=500, stretch=YES)
        tree.column("3", width=550, stretch=YES)
        tree.column("4", width=150, stretch=YES)

        # heading of the table
        tree.heading("#0", text="S.N.", )
        tree.heading("1", text="Created At")
        tree.heading("2", text="Tweets", )
        tree.heading("3", text="Clean Tweets", )
        tree.heading("4", text=" Locations", )

        tree.place(x=0, y=50, width=1020,height=500)
        i = 0
        with open('nonemptydata.csv') as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                i = i + 1
                tweets = row['text']
                clean_tweets=row['Clean Tweets']
                created_at = row['created_at']
                locations = row['Location']
                tree.insert("", i, text=i, values=(created_at, tweets,clean_tweets,locations))
    def calculateSentiment(self):
        f1 = Frame(self.cleanTweetsWindow, bg="red", height=50, width=200)
        f1.pack(side=BOTTOM, anchor=E, padx=20, pady=10)
        def sentiments():
            def getSubjectivity(text):
                return round(TextBlob(text).sentiment.subjectivity,2)
            def getPolarity(text):
                return round(TextBlob(text).sentiment.polarity,2)

            def getAnalysis(score):
                if score < 0:
                    return 'Negative'
                elif score == 0:
                    return 'Neutral'
                else:
                    return 'Positive'
            df = pd.read_csv('nonemptydata.csv')
            df['Subjectivity'] = df['Clean Tweets'].apply(getSubjectivity)
            df['Polarity'] = df['Clean Tweets'].apply(getPolarity)
            df['Analysis'] = df['Polarity'].apply(getAnalysis)
            df.to_csv(r'sentimentCalculatedData.csv', index=False)
            showSentimentPage()

        Button(f1, text="Calculate Sentiments",command=sentiments ).place(x=0, y=0, height=50, width=200, )

class showSentimentPage():
    def __init__(self):
        self.showSentimentWindow=Tk()
        self.showSentimentWindow.geometry("1020x680+160+10")
        self.showSentimentWindow.title("Sentiment of Text")
        self.showSentimentWindow.resizable(0, 0)
        self.showSentiment()
        self.getAnalysis()
        self.showSentimentWindow.mainloop()

    def showSentiment(self):
        Label(self.showSentimentWindow, text="Sentiment of the Text", font=("m 30")).place(x=370, y=5)
        tree = ttk.Treeview(self.showSentimentWindow)

        # number of columns
        tree["columns"] = ("1", "2", "3", "4")
        tree.column("#0", width=40, stretch=YES)
        tree.column("1", width=700, stretch=YES)
        tree.column("2", width=50, stretch=YES)
        tree.column("3", width=50, stretch=YES)
        tree.column("4", width=50, stretch=YES)

        # heading of the table
        tree.heading("#0", text="S.N.", )
        tree.heading("1", text="Clean Tweets")
        tree.heading("2", text="Subjectivity", )
        tree.heading("3", text="Polarity", )
        tree.heading("4", text="Sentiments", )

        tree.place(x=0, y=50, width=1020, height=500)
        i = 0
        with open('sentimentCalculatedData.csv') as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                i = i + 1
                clean_tweets = row['Clean Tweets']
                subjectivity = row['Subjectivity']
                polarity = row['Polarity']
                analysis=row['Analysis']
                tree.insert("", i, text=i, values=(clean_tweets, subjectivity, polarity, analysis))

    def getAnalysis(self):
        f1 = Frame(self.showSentimentWindow,  height=50, width=150)
        f2 = Frame(self.showSentimentWindow,  height=50, width=150)
        f3 = Frame(self.showSentimentWindow, height=50, width=150)
        f4 = Frame(self.showSentimentWindow, height=50, width=150)
        f5 = Frame(self.showSentimentWindow, height=50, width=150)

        f1.place(x=820, y=600)
        f2.place(x=620, y=600)
        f3.place(x=420, y=600)
        f4.place(x=220,y=600)
        f5.place(x=20, y=600)

        def positiveTweets():
            data = pd.read_csv("sentimentCalculatedData.csv")
            positiveTweets=data.loc[data['Analysis'] == 'Positive']
            positiveTweets.to_csv('positiveTweets.csv',index=False)

            Label(self.showSentimentWindow, text="Sentiment of the Text", font=("m 30")).place(x=370, y=5)
            tree = ttk.Treeview(self.showSentimentWindow)

            # number of columns
            tree["columns"] = ("1", "2", "3", "4")
            tree.column("#0", width=40, stretch=YES)
            tree.column("1", width=700, stretch=YES)
            tree.column("2", width=50, stretch=YES)
            tree.column("3", width=50, stretch=YES)
            tree.column("4", width=50, stretch=YES)

            # heading of the table
            tree.heading("#0", text="S.N.", )
            tree.heading("1", text="Clean Tweets")
            tree.heading("2", text="Subjectivity", )
            tree.heading("3", text="Polarity", )
            tree.heading("4", text="Sentiments", )

            tree.place(x=0, y=50, width=1020, height=500)
            i = 0
            with open('positiveTweets.csv') as f:
                reader = csv.DictReader(f, delimiter=',')
                for row in reader:
                    i = i + 1
                    clean_tweets = row['Clean Tweets']
                    subjectivity = row['Subjectivity']
                    polarity = row['Polarity']
                    analysis = row['Analysis']
                    tree.insert("", i, text=i, values=(clean_tweets, subjectivity, polarity, analysis))
        def negativeTweets():
            data = pd.read_csv("sentimentCalculatedData.csv")
            positiveTweets = data.loc[data['Analysis'] == 'Negative']
            positiveTweets.to_csv('negativeTweets.csv', index=False)

            Label(self.showSentimentWindow, text="Sentiment of the Text", font=("m 30")).place(x=370, y=5)
            tree = ttk.Treeview(self.showSentimentWindow)

            # number of columns
            tree["columns"] = ("1", "2", "3", "4")
            tree.column("#0", width=40, stretch=YES)
            tree.column("1", width=700, stretch=YES)
            tree.column("2", width=50, stretch=YES)
            tree.column("3", width=50, stretch=YES)
            tree.column("4", width=50, stretch=YES)

            # heading of the table
            tree.heading("#0", text="S.N.", )
            tree.heading("1", text="Clean Tweets")
            tree.heading("2", text="Subjectivity", )
            tree.heading("3", text="Polarity", )
            tree.heading("4", text="Sentiments", )

            tree.place(x=0, y=50, width=1020, height=500)
            i = 0
            with open('negativeTweets.csv') as f:
                reader = csv.DictReader(f, delimiter=',')
                for row in reader:
                    i = i + 1
                    clean_tweets = row['Clean Tweets']
                    subjectivity = row['Subjectivity']
                    polarity = row['Polarity']
                    analysis = row['Analysis']
                    tree.insert("", i, text=i, values=(clean_tweets, subjectivity, polarity, analysis))
        def neutralTweets():
            data = pd.read_csv("sentimentCalculatedData.csv")
            positiveTweets = data.loc[data['Analysis'] == 'Neutral']
            positiveTweets.to_csv('positiveTweets.csv', index=False)

            Label(self.showSentimentWindow, text="Sentiment of the Text", font=("m 30")).place(x=370, y=5)
            tree = ttk.Treeview(self.showSentimentWindow)

            # number of columns
            tree["columns"] = ("1", "2", "3", "4")
            tree.column("#0", width=40, stretch=YES)
            tree.column("1", width=700, stretch=YES)
            tree.column("2", width=50, stretch=YES)
            tree.column("3", width=50, stretch=YES)
            tree.column("4", width=50, stretch=YES)

            # heading of the table
            tree.heading("#0", text="S.N.", )
            tree.heading("1", text="Clean Tweets")
            tree.heading("2", text="Subjectivity", )
            tree.heading("3", text="Polarity", )
            tree.heading("4", text="Sentiments", )

            tree.place(x=0, y=50, width=1020, height=500)
            i = 0
            with open('positiveTweets.csv') as f:
                reader = csv.DictReader(f, delimiter=',')
                for row in reader:
                    i = i + 1
                    clean_tweets = row['Clean Tweets']
                    subjectivity = row['Subjectivity']
                    polarity = row['Polarity']
                    analysis = row['Analysis']
                    tree.insert("", i, text=i, values=(clean_tweets, subjectivity, polarity, analysis))
        def showAll():
            Label(self.showSentimentWindow, text="Sentiment of the Text", font=("m 30")).place(x=370, y=5)
            tree = ttk.Treeview(self.showSentimentWindow)

            # number of columns
            tree["columns"] = ("1", "2", "3", "4")
            tree.column("#0", width=40, stretch=YES)
            tree.column("1", width=700, stretch=YES)
            tree.column("2", width=50, stretch=YES)
            tree.column("3", width=50, stretch=YES)
            tree.column("4", width=50, stretch=YES)

            # heading of the table
            tree.heading("#0", text="S.N.", )
            tree.heading("1", text="Clean Tweets")
            tree.heading("2", text="Subjectivity", )
            tree.heading("3", text="Polarity", )
            tree.heading("4", text="Sentiments", )

            tree.place(x=0, y=50, width=1020, height=500)
            i = 0
            with open('sentimentCalculatedData.csv') as f:
                reader = csv.DictReader(f, delimiter=',')
                for row in reader:
                    i = i + 1
                    clean_tweets = row['Clean Tweets']
                    subjectivity = row['Subjectivity']
                    polarity = row['Polarity']
                    analysis = row['Analysis']
                    tree.insert("", i, text=i, values=(clean_tweets, subjectivity, polarity, analysis))







        Button(f5, text="Neutral Tweets", command=neutralTweets,activeforeground="#2ecc71").place(x=0, y=0, height=50, width=150, )
        Button(f4, text="Negative Tweets", command=negativeTweets,activeforeground="#2ecc71").place(x=0, y=0, height=50, width=150, )
        Button(f3, text="Positive Tweets",command=positiveTweets,activeforeground="#2ecc71").place(x=0, y=0, height=50, width=150, )
        Button(f1, text="All Data", command=showAll,activeforeground="#2ecc71",).place(x=0, y=0, height=50, width=150, )
        Button(f2, text="Details", command=Details,activeforeground="#2ecc71").place(x=0, y=0, height=50, width=150, )

class trainData():
    def __init__(self):
        self.trainData = Tk()
        self.trainData.geometry("1020x680+160+10")
        self.trainData.title("Training the Data")
        self.trainData.resizable(0, 0)
        self.showAccuracy()
        self.showData()
        # self.getPrediction()
        self.trainData.mainloop()
    def showAccuracy(self):

        data = pd.read_csv("sentimentCalculatedData.csv")

        data_clean = data.copy()
        data_clean['sentiment'] = data_clean['Analysis'].apply(lambda x: 1 if x == 'Negative' else 0)
        data_clean = data_clean.loc[:, ['Clean Tweets', 'sentiment']]
        print(data_clean)
        train, test = train_test_split(data_clean, test_size=0.2, random_state=1)
        self.X_train = train['Clean Tweets'].values
        self.X_test = test['Clean Tweets'].values

        self.y_train = train['sentiment']
        self.y_test = test['sentiment']

        def tokenize(text):
            tknzr = TweetTokenizer()
            return tknzr.tokenize(text)

        def stem(doc):
            return (stemmer.stem(w) for w in analyzer(doc))

        en_stopwords = set(stopwords.words("english"))

        vectorizer = CountVectorizer(
            analyzer='word',
            tokenizer=tokenize,
            lowercase=True,
            ngram_range=(1, 1),
            stop_words=en_stopwords)

        kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        pipeline_svm = make_pipeline(vectorizer,SVC(probability=True, kernel="rbf", class_weight="balanced"))

        self.grid_svm = GridSearchCV(pipeline_svm,
                                param_grid={'svc__C': [0.01, 0.1, 1]},
                                cv=kfolds,
                                scoring="roc_auc",
                                verbose=1,
                                n_jobs=-1)

        a=self.grid_svm.fit(self.X_train, self.y_train)
        b=self.grid_svm.score(self.X_test, self.y_test)



        def report_results(model, X, y):
            pred_proba = model.predict_proba(X)[:, 1]
            pred = model.predict(X)

            auc = roc_auc_score(y, pred_proba)
            acc = accuracy_score(y, pred)
            f1 = f1_score(y, pred)
            prec = precision_score(y, pred)
            rec = recall_score(y, pred)
            result = {'auc': auc, 'f1': f1, 'acc': acc, 'precision': prec, 'recall': rec}
            return result

        self.x=report_results(self. grid_svm.best_estimator_, self.X_test, self.y_test)
        # print(x)
        # print("roc auc","%.2f" % x['auc'])
        # print("acc",x['acc'])
        # print("f1", x['f1'])
        # print("precision",x["precision"])
        # print("recall", x['recall'])

    def showData(self):
        Label(self.trainData, text="Report Results", font=("m 30")).place(x=400, y=10)

        f1 = Frame(self.trainData, background="#2ecc71", )
        f1.place(x=50, y=70)
        Label(f1, text="  ROC ACC  ", background="#2ecc71", foreground="white").pack()
        Label(f1, text="%.2f" %self.x['auc'], background="#2ecc71", foreground="white").pack()

        f2 = Frame(self.trainData, bg="violet")
        f2.place(x=250, y=70)
        Label(f2, text="  Accuracy  ", background="violet", foreground="white").pack()
        Label(f2, text="%.2f" %self.x['acc'], background="violet", foreground="white").pack()

        f3 = Frame(self.trainData, bg="#8e44ad", )
        f3.place(x=450, y=70)
        Label(f3, text="  f1 score  ", background="#8e44ad", foreground="white").pack()
        Label(f3, text="%.2f" %self.x['f1'], background="#8e44ad", foreground="white").pack()

        f4 = Frame(self.trainData, bg="red", )
        f4.place(x=650, y=70)
        Label(f4, text="  Precision  ", background="red", foreground="white").pack()
        Label(f4, text="%.2f" %self.x['precision'], background="red", foreground="white").pack()

        f5 = Frame(self.trainData, bg="#e74c3c", )
        f5.place(x=850, y=70)
        Label(f5, text="   Recall   ", background="#e74c3c", foreground="white").pack()
        Label(f5, text="%.2f" %self.x['recall'], background="#e74c3c", foreground="white").pack()

        Label (self.trainData, text="Predict").place(x=450, y=150)
        predictText = Entry(self.trainData, )
        predictText.place(x=50, y=200, width=700, height=60)

        def prediction():
            predictionText=predictText.get()
            if not predictionText:
                messagebox.showerror("Error", "Please enter the text")


            else:
                x = self.grid_svm.predict([predictionText])
                if (x == [0]):
                    self.positiveLabel=Label(self.trainData, text="The sentence is Positive", foreground="green", font="m 30 bold").place(
                        x=350, y=270)
                else:
                    self.negativeLabel=Label(self.trainData, text="The sentence is Negative", foreground="red", font="m 30 bold").place(
                        x=350, y=270)




        Button(self.trainData, text="Predict", command=prediction,padx=40, pady=20).place(x=800, y=200)

        def rocCurve():
            def get_roc_curve(model, X, y):
                pred_proba = model.predict_proba(X)[:, 1]
                fpr, tpr, _ = roc_curve(y, pred_proba)
                return fpr, tpr

            roc_svm = get_roc_curve(self.grid_svm.best_estimator_, self.X_test, self.y_test)
            fpr, tpr = roc_svm
            plt.figure(figsize=(14, 8))
            plt.plot(fpr, tpr, color="red")
            plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Roc curve')
            plt.show()

        def leaningCurve():
            train_sizes, train_scores, test_scores = learning_curve(self.grid_svm.best_estimator_, self.X_train, self.y_train, cv=5,
                                                                    n_jobs=-1,
                                                                    scoring="roc_auc",
                                                                    train_sizes=np.linspace(.1, 1.0, 10),
                                                                    random_state=1)

            def plot_learning_curve(X, y, train_sizes, train_scores, test_scores, title='', ylim=None, figsize=(14, 8)):
                plt.figure(figsize=figsize)
                plt.title(title)
                if ylim is not None:
                    plt.ylim(*ylim)
                plt.xlabel("Training examples")
                plt.ylabel("Score")

                train_scores_mean = np.mean(train_scores, axis=1)
                train_scores_std = np.std(train_scores, axis=1)
                test_scores_mean = np.mean(test_scores, axis=1)
                test_scores_std = np.std(test_scores, axis=1)
                plt.grid()

                plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                                 train_scores_mean + train_scores_std, alpha=0.1,
                                 color="r")
                plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
                plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                         label="Training score")
                plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                         label="Cross-validation score")

                plt.legend(loc="lower right")
                return plt

            plot_learning_curve(self.X_train, self.y_train, train_sizes,
                                train_scores, test_scores, ylim=(0.7, 1.01), figsize=(14, 6))
            plt.show()





        Button(self.trainData, text="ROC Curve", command=rocCurve,padx=40, pady=20).place(x=50,y=360)
        Button(self.trainData, text="Learning Curve", command=leaningCurve, padx=40, pady=20).place(x=250, y=360)


class Details():
    def __init__(self):
        self.showDetails=Tk()
        # self.showDetails.geometry("1020x680+160+10")
        self.showDetails.title("Details of the tweets")
        # self.showDetails.resizable(0, 0)
        self.tweetDetails()

        self.showDetails.mainloop()

    def tweetDetails(self):
        data = pd.read_csv('sentimentCalculatedData.csv')
        data1 = pd.read_csv('nonemptydata.csv')
        # Create a DataFrame object
        dfObj = pd.DataFrame(data1, columns=['Location'])
        totalTweets=len(data1)
        tweetsWithoutLocation=dfObj.isnull().sum().sum()
        tweetsWithLocation=totalTweets-tweetsWithoutLocation

        ptweets = data[data.Analysis == 'Neutral']
        ptweets = ptweets['Clean Tweets']
        positiveTweets=ptweets.shape[0]

        ptweets = data[data.Analysis == 'Negative']
        ptweets = ptweets['Clean Tweets']
        negativeTweets = ptweets.shape[0]

        neutralTweets=len(data)-positiveTweets-negativeTweets

        def locationGraph():
            objects = ('With Location', 'Without Location')
            y_pos = np.arange(len(objects))
            performance = [tweetsWithLocation,tweetsWithoutLocation]

            plt.bar(y_pos, performance, align='center', alpha=0.5)
            plt.xticks(y_pos, objects)
            plt.ylabel('Number of Tweets')
            plt.title('Total number of tweets with location and without location')
            plt.show()




        print("Total number of tweets",totalTweets)
        print("Total number of tweets without location",tweetsWithoutLocation)
        print("Total number of tweets with location",tweetsWithLocation)

        print("Total number of positive Tweets",positiveTweets)
        print("Total number of negative Tweets",negativeTweets )
        print("Total number of neutral Tweets", neutralTweets)


        Label(self.showDetails, text="Total number of tweets:   "+str(totalTweets), font=("w 30 bold")).pack(anchor=W)
        Label(self.showDetails, text="Total number of tweets without location:   " + str(tweetsWithoutLocation), font=("w 30 bold")).pack(anchor=W)
        Label(self.showDetails, text="Total number of tweets with location:   " + str(tweetsWithLocation),
              font=("w 30 bold")).pack(anchor=W)
        Label(self.showDetails, text="Total number of positive tweets:   " + str(positiveTweets), font=("w 30 bold")).pack(anchor=W)
        Label(self.showDetails, text="Total number of negative tweets:   " + str(negativeTweets), font=("w 30 bold")).pack(anchor=W)
        Label(self.showDetails, text="Total number of neutral tweets:   " + str(neutralTweets), font=("w 30 bold")).pack(anchor=W)


        # Frame(self.showDetails,height=0).pack()
        def bargraph():
            df = pd.read_csv('sentimentCalculatedData.csv')
            plt.figure(figsize=(8, 7))
            # Show the value counts
            df['Analysis'].value_counts()


            # Plotting and visualizing the counts
            plt.title('Sentiment Analysis')
            plt.xlabel('Sentiment')
            plt.ylabel('Counts')
            df['Analysis'].value_counts().plot(kind='bar')
            plt.show()
        def wordcloud():
            # word cloud visualization
            df = pd.read_csv('nonemptydata.csv')
            allWords = ' '.join([twts for twts in df['Clean Tweets']])
            wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(allWords)
            plt.imshow(wordCloud, interpolation="bilinear")
            plt.axis('off')
            plt.show()

        def subvspol():
            df = pd.read_csv('sentimentCalculatedData.csv')
            plt1.figure(figsize=(8, 6))
            for i in range(0, df.shape[0]):
                plt1.scatter(df["Polarity"][i], df["Subjectivity"][i], color='Blue')
            plt1.title('Sentiment Analysis')
            plt1.xlabel('Polarity')
            plt1.ylabel('Subjectivity')
            plt1.show()


        def locationanalysis():
            data = pd.read_csv('sentimentCalculatedData.csv')
            # plt.figure(figsize=(8, 7))
            # plt.title("Sentiment of tweets on different date")
            data.groupby(['Location', 'Analysis']).Analysis.count().unstack().plot(kind='bar')
            plt2.show()

        def dateanalysis():
            data = pd.read_csv('sentimentCalculatedData.csv')
            # plt.figure(figsize=(8, 7))
            # plt.title("Sentiment of tweets on different date")
            data.groupby(['created_at', 'Analysis']).Analysis.count().unstack().plot(kind='bar')
            plt2.show()

        Button(self.showDetails, command=locationGraph, text="Location Tweets", font=("w 20 bold")).pack(ipadx=20, ipady=10,
                                                                                                padx=20,side=LEFT)
        Button(self.showDetails, command=bargraph, text="Bar Graph",font=("w 20 bold")).pack(ipadx=20, ipady=10,padx=20,side=LEFT,)
        Button(self.showDetails, command=wordcloud, text="Word Cloud", font=("w 20 bold")).pack(ipadx=20, ipady=10,padx=20,side=LEFT)
        Button(self.showDetails, command=subvspol, text="Sub. Vs Pol.", font=("w 20 bold")).pack(ipadx=20, ipady=10,padx=20,side=LEFT)
        Button(self.showDetails, command=locationanalysis, text=" Analysis", font=("w 20 bold")).pack(ipadx=20, ipady=10,padx=20,side=LEFT)
        Button(self.showDetails, command=dateanalysis, text="Date Analysis", font=("w 20 bold")).pack(ipadx=20,
                                                                                                      ipady=10, padx=20,
                                                                                                     side=)
        Button(self.showDetails, command=trainData, text="Train Data", font=("w 20 bold")).pack(ipadx=20, ipady=10,
                                                                                             padx=20,side=TOP)










s=Details()