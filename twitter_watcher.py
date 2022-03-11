from twitter_api import search as twitter_search
import pandas as pd
from NLP import NLP
from difflib import SequenceMatcher
import time
import numpy as np
from threading import Thread

# csv columns: event_id, event_type, event_location, event_status, recent_tweets, relevant_hashtags, is_active

class Twitter_Watcher(Thread):
    def __init__(self, csv_path, sleep_time=60, default_hashtag="#TwitterDisasterWatcher"):
        super().__init__()
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.sleep_time = sleep_time
        self.default_hashtag = default_hashtag
        self.nlp = NLP(
            "./models/tf_bert/model_weights_9.h5",
            "./models/spacy_ner_cpu/model-best"
        )
    
    def run_event_search(self, bert_thres=0.5):
        rows = self.df[self.df["is_active"] == "yes"]

        for i in range(len(rows.index)):
            row = rows.iloc[i]
            row_dict = row.to_dict()
            hashtag_list = row["relevant_hashtags"].split("|")
            print(hashtag_list)

            query = " OR ".join(hashtag_list)
            query = f"({query})"

            tweets = twitter_search(query)

            temp_list = [tweet['text'] for tweet in tweets if tweet['lang'] == "en"]
            bert_results = self.nlp.process_bert(temp_list)

            relevent_tweets = []
            for j, item in enumerate(bert_results):
                try:
                    prob = item[0][0]
                except:
                    prob = item[0]
                if prob > bert_thres:
                    relevent_tweets.append(tweets[j])
                
            tweet_ids = []
            tweet_texts = [tweet['text'] for tweet in relevent_tweets]
            print(tweet_texts)

            # we have to update tweet list, and status

            spacy_results = self.nlp.process_spacy(tweet_texts)
            new_status = ""
            for j, item in enumerate(spacy_results):
                for word in item.ents:
                    print(word.label_, word.text)
                    print("STATUS " in str(word.label_))
                    if word.label_ == "STATUS ":
                        new_status += f"|{word.text}"            
                        tweet_ids.append(str(relevent_tweets[j]["id"]))

            print(f"new status: {new_status}")
            print(f"tweet ids: {tweet_ids}")

            self.df["event_status"][i] = new_status
            self.df["recent_tweets"][i] = "|".join(tweet_ids)
            # self.df.iloc[i] = row
            print(self.df)
            self.df.to_csv(self.csv_path, index=False)

    def reload_data(self):
        self.df = pd.read_csv(self.csv_path)

    def run_default_search(self):
        active_events = self.df[self.df["is_active"] == "yes"]

        tweets = twitter_search(self.default_hashtag)
        temp_list = [tweet['text'] for tweet in tweets if tweet["lang"] == "en"]

        bert_results = self.nlp.process_bert(temp_list)
        print(bert_results)
        relevent_tweets = []
        for j, item in enumerate(bert_results):
            # print(len(item[0]))
            try:
                prob = item[0][0]
            except:
                prob = item[0]
            if prob > 0.2:
                relevent_tweets.append(tweets[j])

        tweet_ids = []
        tweet_texts = [tweet['text'] for tweet in relevent_tweets]
        spacy_results = self.nlp.process_spacy(tweet_texts)
        for item in spacy_results:
            context_event = ""
            for word in item.ents:
                print(word.label_, word.text)
                if word.label_ == "LOCATION":
                    print(word.text)
                    similarity_list = np.array([similar(word.text, location) for location in active_events["event_location"]])
                    similarityindex = np.argmax(similarity_list)

                    if similarity_list[similarityindex] > 0.5:
                        context_event = active_events["event_id"][similarityindex]

                if context_event != "" and word.label_ == "STATUS ":
                    print(word.text)
                    active_events.loc[active_events["event_id"] == context_event, "event_status"] += f"|{word.text}"     
        
        # update the df
        self.df.loc[self.df["is_active"] == "yes"] = active_events
        print(self.df)
        self.df.to_csv(self.csv_path, index=False)

    def refrest_database(self):
        self.df=pd.read_csv(self.csv_path) 

    def run(self):
        while True:
            self.run_event_search()
            # self.run_default_search()
            time.sleep(self.sleep_time)

def get_hashtags(tweet):
    return [hashtag.tag for hashtag in tweet["entities"]["hashtags"]]

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

if __name__ == "__main__":
    print(twitter_search("#covid19"))
