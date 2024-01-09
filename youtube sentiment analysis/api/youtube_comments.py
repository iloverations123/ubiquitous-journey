# api/analyze.py
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Load the saved model
loaded_model = tf.keras.models.load_model("C:\\Users\\benel\\Downloads\\youtube sentiment analysis")

# Function to preprocess input text
def preprocess_input_text(input_text):
    input_sequences = tokenizer.texts_to_sequences([input_text])
    input_padded = pad_sequences(input_sequences, maxlen=118)  # Adjust with your actual sequence length
    return input_padded

# Function to predict sentiment
def predict_sentiment(input_text):
    preprocessed_input = preprocess_input_text(input_text)
    predictions = loaded_model.predict(preprocessed_input)
    predicted_label_index = tf.argmax(predictions, axis=1).numpy()[0]
    numbers_to_words = {0: "Negative", 1: "Positive"}  # Adjust based on your label mapping
    predicted_sentiment = numbers_to_words[predicted_label_index]
    return predicted_sentiment

def handler(request):
    try:
        data = request.get_json()
        youtube_url = data.get('url')

        # Download YouTube comments
        downloader = YoutubeCommentDownloader()
        comments = downloader.get_comments_from_url(youtube_url, sort_by=SORT_BY_RECENT)

        # Perform sentiment analysis on each comment
        results = []
        for i, comment in enumerate(comments):
            sentiment_score = predict_sentiment(comment['text'])
            results.append({'comment': comment['text'], 'sentiment': sentiment_score})

        return results
    except Exception as e:
        return {"error": str(e)}
