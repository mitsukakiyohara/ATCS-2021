# US Elections 2020 Sentiment Analysis 
# Name: Mitsuka Kiyohara
# Block B | AT CS 

#Import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import geopandas as gpd
import plotly.graph_objects as go
from matplotlib import cm, dates
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from shapely.geometry import Point, Polygon

#PT 1. DATA CLEANING 
#Preprocess Trump Dataset for analysis
df = pd.read_csv("dataset/hashtag_donaldtrump.csv", lineterminator='\n')
df_trump = df.drop(columns=['tweet_id', 'source', 'user_screen_name', 'user_description','state_code', 'collected_at', 'user_location', 'city'])

#Removing tweets with NaN values for user_location 
df_trump = df_trump.dropna(axis='rows', thresh=10)

#Preprocess Biden Dataset for analysis
df = pd.read_csv("dataset/hashtag_joebiden.csv", lineterminator='\n')
df_biden = df.drop(columns=['tweet_id', 'source', 'user_screen_name', 'user_description', 'state_code', 'collected_at', 'city', 'user_location'])

#Removing tweets with NaN values for user_location 
df_biden = df_biden.dropna(axis='rows', thresh=10)

#Creating a test dataframe (5000 tweets for fater processing time)
test_trump = df_trump.sample(10000)
test_biden = df_biden.sample(10000)

#PT 2. ANALYSIS 
#Find Subjectivity of Polarity for each Tweet using TextBlob for NLP 
tweets_trump = [TextBlob(desc) for desc in test_trump['tweet']]
tweets_biden = [TextBlob(desc) for desc in test_biden['tweet']]

#Adding sentiment metrics to the dataframe
test_trump['polarity'] = [b.polarity for b in tweets_trump]
test_trump['subjectivity'] = [b.subjectivity for b in tweets_trump]

test_biden['polarity'] = [b.polarity for b in tweets_biden]
test_biden['subjectivity'] = [b.subjectivity for b in tweets_biden]

#Setting polarity values to each sentiment 
test_trump.loc[test_trump.polarity > 0,'sentiment'] = 'positive'
test_trump.loc[test_trump.polarity == 0,'sentiment'] = 'neutral'
test_trump.loc[test_trump.polarity < 0,'sentiment'] = 'negative'

test_biden.loc[test_biden.polarity > 0,'sentiment'] = 'positive'
test_biden.loc[test_biden.polarity == 0,'sentiment'] = 'neutral'
test_biden.loc[test_biden.polarity < 0,'sentiment'] = 'negative'

#Data Visualization 1: World Map of all Tweets 
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
crs = {'init': 'EPSG:4326'}

tmp=pd.concat([df_biden[['lat','long']].copy(), df_trump[['lat','long']].copy()])
tmp = tmp.dropna()
geometry = [Point(xy) for xy in zip(tmp['long'],tmp['lat'])]
geo_df = gpd.GeoDataFrame(tmp, crs=crs, geometry = geometry)

fig, ax = plt.subplots(1,figsize=(16,8), facecolor='lightblue')
world = world[world.name != "Antarctica"]
world.plot(ax=ax, cmap='OrRd', edgecolors='black')
geo_df.plot(ax=ax, markersize=1, color='b', marker='o')
ax.axis('off')
#plt.show()


#Data Visualization 2: Word Clouds of Tweets
#Looking at common words found in Trump Tweets
plt.subplots(1,1, figsize=(9,9))
wc_b = WordCloud(stopwords=STOPWORDS, 
                 background_color="white", max_words=2000,
                 max_font_size=256, random_state=42,
                 width=1600, height=1600)
wc_b.generate(str(test_trump.dropna()))
plt.imshow(wc_b, interpolation="bilinear")
plt.axis('off')
#plt.show()

#Looking at Trump Tweets just from the United States 
text1 = test_trump.loc[test_trump['country'] == 'United States of America']['tweet']
plt.subplots(1,1, figsize=(9,9))
wc_t = WordCloud(stopwords=STOPWORDS, 
                 background_color="black", max_words=2000,
                 max_font_size=256, random_state=42,
                 width=1600, height=1600)
wc_t.generate(str(text1.dropna()))
plt.imshow(wc_t, interpolation="bilinear")
plt.axis('off')
#plt.show()

#Looking at common words found in Biden Tweets
plt.subplots(1,1, figsize=(9,9))
wc_b = WordCloud(stopwords=STOPWORDS, 
                 background_color="white", max_words=2000,
                 max_font_size=256, random_state=42,
                 width=1600, height=1600)
wc_b.generate(str(test_biden.dropna()))
plt.imshow(wc_b, interpolation="bilinear")
plt.axis('off')
#plt.show()

#Looking at Biden Tweets just from the United States 
text2 = df_biden.loc[df_biden['country'] == 'United States of America']['tweet']
plt.subplots(1,1, figsize=(9,9))
wc_t = WordCloud(stopwords=STOPWORDS, 
                 background_color="black", max_words=2000,
                 max_font_size=256, random_state=42,
                 width=1600, height=1600)
wc_t.generate(str(text2.dropna()))
plt.imshow(wc_t, interpolation="bilinear")
plt.axis('off')
#plt.show()

#Data Visualization 3: Mean Sentiment in US States over 14 Days
#Create 52 state set
states = set(test_trump.loc[test_trump['country'] == 'United States of America']['state'].dropna())
states.remove('District of Columbia')
#tates.remove('Northern Mariana Islands')

#Create feature to allow masking of data and then mask data for votable states
test_biden['voting_rights'] = test_biden['state'].apply(lambda x: 'Yes' if x in states else 'No')
test_trump['voting_rights'] = test_trump['state'].apply(lambda x: 'Yes' if x in states else 'No')
sent_t = test_trump.loc[test_trump['voting_rights'] == 'Yes']
sent_b = test_biden.loc[test_biden['voting_rights'] == 'Yes']

#Only grab data from the last 14 days 
sent_t['created_at_time'] = pd.to_datetime(sent_t['created_at'], utc=True)
sent_b['created_at_time'] = pd.to_datetime(sent_b['created_at'], utc=True)
state_t = sent_t.loc[sent_t['created_at_time'] > max(sent_t['created_at_time']) - timedelta(14)]
state_b = sent_b.loc[sent_b['created_at_time'] > max(sent_b['created_at_time']) - timedelta(14)]

state_b_mean = state_b.groupby('state')['subjectivity'].mean().reset_index()
state_t_mean = state_t.groupby('state')['subjectivity'].mean().reset_index()

#Only grab data from the first 14 days 
state_bp = sent_b.loc[sent_b['created_at_time'] < min(sent_b['created_at_time']) + timedelta(14)]
state_tp = sent_t.loc[sent_t['created_at_time'] < min(sent_t['created_at_time']) + timedelta(14)]
state_bp_mean = state_bp.groupby('state')['subjectivity'].mean().reset_index()
state_tp_mean = state_tp.groupby('state')['subjectivity'].mean().reset_index()

#Create new dataframe states_sent
states_sent = pd.DataFrame({'state':state_b_mean['state'],
                          'biden1':state_b_mean['subjectivity'],
                          'trump1':state_t_mean['subjectivity'],
                          'biden2':state_bp_mean['subjectivity'],
                          'trump2':state_tp_mean['subjectivity'],})

#Line/Scatter Plot for Trump
fig, ax = plt.subplots(2,1, figsize = (22,20), gridspec_kw = {'hspace':0.05})
lineax = ax[0]
sns.lineplot(x='state', y='trump1', color='red', data=states_sent, ax=lineax, label='Trump (L14D)').set_title('Mean Sentiment Score in US states over 14 Days')
sns.scatterplot(x='state', y='trump1', color='red', data=states_sent, ax=lineax)
sns.lineplot(x='state', y='trump2', color='lightgrey', data=states_sent, ax=lineax, label='Trump (F14D)')
sns.scatterplot(x='state', y='trump2', color='lightgrey', data=states_sent, ax=lineax)
lineax.set_ylim([-1, 1])
lineax.set_ylabel('Mean subjectivity score (for the last 14D)')
lineax.set_xlabel('')
plt.xticks(rotation=90)
lineax.axhline(y=0, color='k', linestyle='-')
lineax.axhline(y=0.05, color='lightgrey', linestyle='-')
lineax.axhline(y=-0.05, color='lightgrey', linestyle='-')
lineax.axes.get_xaxis().set_ticks([])
lineax.spines['right'].set_visible(False)
lineax.spines['top'].set_visible(False)
lineax.spines['bottom'].set_visible(False)

#Line/Scatter Plot for Biden
lineax=ax[1]
sns.lineplot(x='state', y='biden1', color='blue', data = states_sent, ax = lineax, label ='Biden (L14D)')
sns.scatterplot(x='state', y='biden1', color='blue', data=states_sent, ax=lineax)
sns.lineplot(x='state', y = 'biden2', color = 'lightgrey', data = states_sent, ax = lineax, label = 'Biden (F14D)')
sns.scatterplot(x='state', y='biden2', color='lightgrey', data=states_sent, ax=lineax)
lineax.set_ylim([-1, 1])
lineax.set_ylabel('Mean subjectivity score (for the last 14 days)')
lineax.set_xlabel('')
plt.xticks(rotation=90)
lineax.axhline(y = 0, color ='k', linestyle = '-')
lineax.axhline(y = 0.05, color ='lightgrey', linestyle = '-')
lineax.axhline(y = -0.05, color='lightgrey', linestyle = '-')
lineax.spines['right'].set_visible(False)
lineax.spines['top'].set_visible(False)
#plt.show()


#Data Visualization 4: Average Polarity over Time
#For datetime graph purposes, change datetimes to month-date format
test_trump['created_at_time'] = pd.to_datetime(test_trump['created_at'], utc=True)
test_biden['created_at_time'] = pd.to_datetime(test_biden['created_at'], utc=True)

test_trump['created_at_md'] = test_trump['created_at_time'].dt.strftime('%m-%d')
test_biden['created_at_md'] = test_biden['created_at_time'].dt.strftime('%m-%d')

trump_groupby_date_mean = test_trump.groupby(['created_at_md', 'polarity'], as_index=False).mean()
biden_groupby_date_mean = test_biden.groupby(['created_at_md', 'polarity'], as_index=False).mean()

#Create new dataframe datewise_polarity
datewise_polarity = pd.DataFrame(columns=["Date", "Trump Polarity", "Biden Polarity"])
dates = trump_groupby_date_mean.created_at_md

datewise_polarity["Date"] = dates
datewise_polarity["Trump Polarity"] = trump_groupby_date_mean.polarity
datewise_polarity["Biden Polarity"] = biden_groupby_date_mean.polarity
datewise_polarity.set_index("Date", inplace=True)

#Create plot
plt.figure(figsize=(16,6))
plt.title("Average Polarity over Time")
plt.xlabel("Time")
plt.ylabel("Average Polarity")
sns.lineplot(data=datewise_polarity, ci=None, palette=['r', 'b'], dashes=False)
#plt.show()

#Data visualization 5: Sentiment Analysis 
figtrump = px.scatter(test_trump, x="created_at", # date on the x axis
               y="polarity", # sentiment on the y axis
               hover_data=["country", "user_name"], # data to show on hover
               color_discrete_sequence=["lightseagreen", "indianred", "mediumpurple"], # colors to use
               color="sentiment", # represent each sentiment as different color
               size="subjectivity", # the more votes, the bigger the circle
               size_max=10, # not too big (cap size)
               labels={"polarity": "Tweet positivity", "created_at": "Date Tweet was posted"}, # axis names
               title="Trump-Related Tweets Analysis", # title of figure
          )
figtrump.show()

figbiden = px.scatter(test_biden, x="created_at", # date on the x axis
               y="polarity", # sentiment on the y axis
               hover_data=["country", "user_name"], # data to show on hover
               color_discrete_sequence=["lightseagreen", "indianred", "mediumpurple"], # colors to use
               color="sentiment", # represent each sentiment as different color
               size="subjectivity", # the more votes, the bigger the circle
               size_max=10, # not too big (cap size)
               labels={"polarity": "Tweet positivity", "created_at": "Date Tweet was posted"}, # axis names
               title="Biden-Related Tweets Analysis", # title of figure
          )
figbiden.show()

#Data visualization 6: Daily Tweet Count over Time 

#For datetime graph purposes, change datetimes to year-month-date format
test_trump['created_at_ymd'] = test_trump['created_at_time'].dt.strftime('%Y-%m-%d')
test_biden['created_at_ymd'] = test_biden['created_at_time'].dt.strftime('%Y-%m-%d')

#Setting y values 
trump_y = test_trump.groupby(['created_at_ymd', 'sentiment'], as_index=False).count().sort_index()
biden_y = test_biden.groupby(['created_at_ymd', 'sentiment'], as_index=False).count().sort_index()

#Create new dataframe datewise_tweets
datewise_tweets = pd.DataFrame(columns=["Date", "Trump Tweets", "Biden Tweets"])
dates = trump_y.created_at_ymd
datewise_tweets["Date"] = dates
datewise_tweets["Trump Tweets"] = trump_y.created_at
datewise_tweets["Biden Tweets"] = biden_y.created_at
datewise_tweets.set_index("Date",inplace=True)

fig = go.Figure(data=[
    go.Bar(name='Trump', x=dates, y=datewise_tweets["Trump Tweets"]), 
    go.Bar(name='Biden', x=dates, y=datewise_tweets["Biden Tweets"])
])

fig.update_layout(title_text='Weekdays Tweets Joe Biden vs Donald Trump')
fig.update_xaxes(title='Weekdays')
fig.update_yaxes(title='Count of Tweets')
#fig.show()