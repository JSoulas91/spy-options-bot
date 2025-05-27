from textblob import TextBlob
import requests

def get_news_headlines(query="SPY", api_key=None):
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&apiKey={api_key}"
    try:
        response = requests.get(url)
        articles = response.json().get("articles", [])
        return [article["title"] for article in articles]
    except Exception as e:
        print(f"News API error: {e}")
        return []

def analyze_sentiment(headlines):
    sentiment_scores = []
    for headline in headlines:
        blob = TextBlob(headline)
        sentiment_scores.append(blob.sentiment.polarity)
    if not sentiment_scores:
        return 0
    avg_score = sum(sentiment_scores) / len(sentiment_scores)
    return avg_score
