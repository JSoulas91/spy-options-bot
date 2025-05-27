import requests
from textblob import TextBlob

def fetch_reddit_headlines(subreddit="options", limit=20):
    headers = {"User-agent": "Mozilla/5.0"}
    url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
    try:
        response = requests.get(url, headers=headers, timeout=5)
        posts = response.json().get("data", {}).get("children", [])
        return [post["data"]["title"] for post in posts if "title" in post["data"]]
    except Exception as e:
        print(f"Reddit API error: {e}")
        return []

def analyze_social_sentiment(headlines):
    sentiment_scores = []
    for headline in headlines:
        blob = TextBlob(headline)
        sentiment_scores.append(blob.sentiment.polarity)
    if not sentiment_scores:
        return 0
    avg_score = sum(sentiment_scores) / len(sentiment_scores)
    return avg_score
