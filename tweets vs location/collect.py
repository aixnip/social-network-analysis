"""
collect.py
"""
from TwitterAPI import TwitterAPI
import pickle
import sys
import time

consumer_key = 'kfaWV2I2mQnNc4YGWKXQJYYqU'
consumer_secret = 'Ss3HpIvx79Ag8NlZqKvnfNIz57fjQLGorFnTq9rIuI3zSlUTMU'
access_token = '105928326-TyQwpKmgOpJX5mB9pnTNK3HweGQznseRX4ILoPSn'
access_token_secret = '7Pg01ZhjpWfI3LNHUXIxnGgKfNUN9MCxH15gMIuGIGUk0'


def stream_tweets(twitter, location, amount):
    """call twitter streaming api and collect tweets from location for amount
    Arg:
     location - lat long range of tweets location
     amount - amount of tweets wanted

    return
     tweets - array of tweets object
    """
    tweets = []
    done = False
    while not done:
        request = twitter.request('statuses/filter', {'track':'weekend','location':location,'language':'en'})
        if request.status_code != 200:
            print(request.text)
            break
        else:
            for r in request:
                tweets.append(r)
                if len(tweets) % 100 == 0:
                    print('collected %d tweets in %s'%(len(tweets),location))
                if len(tweets) > amount:
                    done = True
                    break
    return tweets

def collect_tweets(twitter):
    tweets_1 = stream_tweets(twitter,'-87,30,-81,25', 300)
    pickle.dump(tweets_1, open('tweets_1.pkl', 'wb'))
    print('sleep for 1 minite')
    time.sleep(61 * 1)
    tweets_2 = stream_tweets(twitter,'-124,49,-117,46', 300)
    pickle.dump(tweets_2, open('tweets_2.pkl', 'wb'))
    return tweets_1, tweets_2

def collect_test_tweets(twitter):
    tweets_1 = stream_tweets(twitter,'-87,30,-81,25', 500)
    pickle.dump(tweets_1, open('tweets_1_test.pkl', 'wb'))
    print('sleep for 1 minite')
    time.sleep(61 * 1)
    tweets_2 = stream_tweets(twitter,'-124,49,-117,46', 500)
    pickle.dump(tweets_2, open('tweets_2_test.pkl', 'wb'))
    return 

def collect_user_network(twitter, tweets, filename):
    #empty the user file
    open(filename,'w').close()
    #download user data, this may take hours
    users = {}
    for t in tweets:
        user_id = t['user']['id']
        if user_id not in users.keys():       
            for i in range(5):
                req = twitter.request("friends/ids", {"user_id":user_id})
                if req.status_code == 200:
                    users[user_id] = list(req)
                    break
                else:
                    pickle.dump(users, open(filename, 'wb'))
                    print('Got error %s \nsleeping for 15 minutes.' % req.text)
                    sys.stderr.flush()
                    time.sleep(61 * 15)     
        if len(users.items()) % 20 == 0:
            print('collected user data for %d users'%len(users.items()))
    pickle.dump(users, open(filename, 'wb'))
    return users

def main():
    twitter = TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)
    tweets_ny, tweets_sf = collect_tweets(twitter)
    total_tweets = []
    total_tweets.extend(tweets_ny)
    total_tweets.extend(tweets_sf)
    collect_test_tweets(twitter)
    user_data = collect_user_network(twitter, total_tweets, 'network.pkl')

    print('done collecting data')
    print('%d tweets'%len(total_tweets))
    print('user data for %d users'%len(user_data.items()))


if __name__ == '__main__':
    main()
