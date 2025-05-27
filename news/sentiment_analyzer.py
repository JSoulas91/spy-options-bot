from textblob import TextBlob

def analyze_sentiment(text):
    """
    Analyzes sentiment using TextBlob.
    Returns a polarity score between -1 and 1.
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity
