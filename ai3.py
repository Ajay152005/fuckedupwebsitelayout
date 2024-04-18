from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np


#load the sentiment analysis pipeline
sentiment_classifier = pipeline("sentiment-analysis")

#function to analyze sentiment of a text

def analyze_sentiment_batch(texts):
    results = sentiment_classifier(texts)
    return results

def visualize_sentiment(results):
    labels = [result['label'] for result in results]
    scores = [result['score'] for result in results]

    x = np.arange(len(labels))
    plt.bar(x, scores, color=['green' if label == 'POSITIVE' else 'red' for label in labels])
    plt.xlabel('Text')
    plt.ylabel('Sentiment Score')
    plt.xticks(x, labels)
    plt.title('Sentiment Analysis Results')
    plt.show()


# Test the sentiment analysis function
text = [
    "I love this movie, it's amazing!",
    "This product is terrible, I regret buying it.",
    "The service at this restaurant was excellent!"
]
results = analyze_sentiment_batch(text)
visualize_sentiment(results)