"""""""""
Twitter Data Fetching Program!
Save your credentials in the same folder as the program
Make sure to use all the different functions I have presented!
Install all the modules below
Scroll down to If name statement - to enter a twitter username!
By Zain Iqbal
Computer Science A level Project
Originally coded in Pycharm

"""""""""



from tweepy import API
from tweepy import Cursor
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

from textblob import TextBlob

import twitter_credentials

import numpy as np    # allow us to refer to anything from the numpy library by using the dot operator
import pandas as pd
import re
import matplotlib.pyplot as plt


name = (input("hello, please enter your name"))
print("Hello", name)

#twitter credentials


from twitter_credentials import consumer_secret, consumer_key, access_token, access_secret


class TwitterClient():

    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)

        self.twitter_user = twitter_user # instaniate, allows user to specifiy a user to get timeline tweets from. default arguement is none, defaults to u

    def get_twitter_client_api(self):
        return self.twitter_client #new function that allows us to interface with this api and extract data from tweets



    def get_user_timeline_tweets(self, num_tweets): # how many tweets we want to extract or share
        tweets = [] # list
        for tweet in Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items(num_tweets): # import cursor, class that allows us to get user timeline tweets
            tweets.append(tweet) #loop thru certain num of tweets and return to user, gets your own timeline tweets, tweets is the var stores the list
        return tweets


    def get_friend_list(self, num_friends):#determine how many friends are fetched
        friend_list = [] # defines list for given user ( no specification of user)
        for friend in Cursor(self.twitter_client.friends, id=self.twitter_user).items(num_friends): # grabs id of friends, starts a loop
            friend_list.append(friend) #friends
        return friend_list


    def get_home_timeline_tweets(self, num_tweets): #twitter homepage tweet fetch for given user/yourself
        home_timeline_tweets = [] # list
        for tweet in Cursor(self.twitter_client.home_timeline, id=self.twitter_user).items(num_tweets):
            home_timeline_tweets.append(tweet)
        return home_timeline_tweets


    #these functions all have a similar flavor as to how they work, have a look and run below in the constructor!



class TwitterAuthenticator():
    def authenticate_twitter_app(self):
        auth = OAuthHandler(consumer_key, consumer_secret)  # authentication
        auth.set_access_token(access_token, access_secret)
        return auth

#abstract this functionality, authenticate for other purposes, new class for authentication
# allows authentication of classes



#twitter streamer

class TwitterStreamer():
    
    #Class for streaming and processing live tweets (from above)
    
    def __init__(self):
        self.twitter_authenticator = TwitterAuthenticator()

    def stream_tweets(self, fetched_tweets_filename, hash_tag_list):
        # This handles Twitter authentication and the connection to Twitter Streaming API
        listener = TwitterListener(fetched_tweets_filename)
        auth = self.twitter_authenticator.authenticate_twitter_app()
        stream = Stream(auth, listener)

        # This line filter Twitter Streams to capture data by the keywords:
        stream.filter(track=hash_tag_list)




class TweetAnalyzer():
    """
     analyzing and categorizing content from tweets.
    """

    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
    # removing special characters from string and hyperlinks and returning the clean tweet

    def analyze_sentiment(self, tweet):#taken from textblob
        analysis = TextBlob(self.clean_tweet(tweet)) #object anaylsing the clean tweet

        if analysis.sentiment.polarity > 0: #anaylsis textblob provides, uses sentiment engine, is it pos or neg
            return 1 #indicates positive tweet
        elif analysis.sentiment.polarity == 0:
            return 0 #indicates neutral tweet
        else:
            return -1 #indicates the tweet is negative.





    def tweets_to_data_frame(self, tweets):#function that allows analysis of tweets and categorizes content from tweets
        df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['tweets'])# these variables are precreated, refer to tweepy documentation for further info
        # looping through every single tweet, extract the text from the text of the tweet

        df['id'] = np.array([tweet.id for tweet in tweets])
        df['len'] = np.array([len(tweet.text) for tweet in tweets])
        df['date'] = np.array([tweet.created_at for tweet in tweets])
        df['source'] = np.array([tweet.source for tweet in tweets])
        df['likes'] = np.array([tweet.favorite_count for tweet in tweets])
        df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])

        return df


if __name__ == '__main__':

    twitter_client = TwitterClient()
    tweet_analyzer = TweetAnalyzer()

    api = twitter_client.get_twitter_client_api()

    tweets = api.user_timeline(screen_name="muftimenk", count=1000000)# what user do you want to grab tweets from

    # print(dir(tweets[0]))
    # print(tweets[0].retweet_count)

    df = tweet_analyzer.tweets_to_data_frame(tweets) # creates dataframe based on function which is based on class
    df['sentiment'] = np.array([tweet_analyzer.analyze_sentiment(tweet) for tweet in df['tweets']])

    # Get average length over all tweets:
    print("average length of tweet from:",np.mean(df['len']))

    # Get the number of likes for the most liked tweet:
    print("max likes of chosen user: ", np.max(df['likes']))

    # Get the number of retweets for the most retweeted tweet:
    print("max retweets of user:" , np.max(df['retweets']))

print(df.head(10000))
"""""""""
#Time Series

time_likes = pd.Series(data=df['likes'].values, index=df['date']) #created a times series object using panda, plotting user data
time_likes.plot(figsize=(16, 4), color='r')
plt.show()



time_retweets = pd.Series(data=df['retweets'].values, index=df['date'])
time_retweets.plot(figsize=(16, 4), color='r')
plt.show()
"""""""""


 #Layered Time Series - visualisation

time_likes = pd.Series(data=df['likes'].values, index=df['date'])
time_likes.plot(figsize=(14,4), label="likes", legend=True) #box that shows what line corresponds to what label

time_retweets = pd.Series(data=df['retweets'].values, index=df['date']) # same thing for retweets
time_retweets.plot(figsize=(14,4), label="retweets", legend=True) #plot for retweets
plt.show() # see both lines together



    # Authenticate using config.py and connect to Twitter Streaming API.
    #hash_tag_list = ["donal trump", "hillary clinton", "barack obama", "bernie sanders"]
    #fetched_tweets_filename = "tweets.txt"

    #twitter_client = TwitterClient('pycon')# user specification, follow the @, extracts timeline tweet from another user
    #print(twitter_client.get_user_timeline_tweets(1)) #prints the function just created

#    twitter_streamer = TwitterStreamer() # method that we created above, file name and list of keywords
#    twitter_streamer.stream_tweets(fetched_tweets_filename, hash_tag_list)


