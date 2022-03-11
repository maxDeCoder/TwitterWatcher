import tweepy
from tweepy import Client
import json

API_creds = json.load(open('./config/tweepy_config.json')) 

consumer_key = API_creds['consumer_key']
consumer_secret = API_creds['consumer_secret']
bearer_token = API_creds['bearer_token']
access_token = API_creds['access_token']
access_token_secret = API_creds['access_token_secret']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit = True)

client = Client(bearer_token, consumer_key, consumer_secret, access_token, access_token_secret, wait_on_rate_limit=True)

def search(query):
    tweets = client.search_recent_tweets(
                                        query, 
                                        max_results=10, 
                                        tweet_fields=[
                                            "id",
                                            "created_at",
                                            "text",
                                            "geo",
                                            "author_id",
                                            "lang",
                                            "entities",
                                        ])

    return [dict(tweet) for tweet in tweets.data]