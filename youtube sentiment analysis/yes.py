import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle

file_path = 'training.csv'
df = pd.read_csv(file_path)

df.columns = ['sentiment', 'id', 'date', 'query', 'user_id', 'text']

numbers_to_words = {0: "Negative", 4: "Positive"}

def label_decoder(label):
    return numbers_to_words[label]

df['sentiment'] = df['sentiment'].map(label_decoder) # basically converts all the values into positive or negative with map()   

X = df['text'].tolist()
Y = df['sentiment'].tolist()


# Convert text to integers
tokenizer = Tokenizer() # the token converts text data into numerical sequences
tokenizer.fit_on_texts(X) # ok basically the token creates its own vocab here

max_sequence_length = max(len(seq) for seq in tokenizer.texts_to_sequences(X))

print(f"Max Sequence Length: {max_sequence_length}")

X_sequences = tokenizer.texts_to_sequences(X) # converts the text to a sequence of integers with the vocab learnt from above
X_padded = pad_sequences(X_sequences) # ensures uniform length and makes it into an array


# Convert labels to integers
label_to_index = {label: index for index, label in enumerate(set(Y))} # dictionary of index: sentiment (positive or negative)
Y_indices = np.array([label_to_index[label] for label in Y]) # basically makes it into a dictionary



X_train, X_test, y_train, y_test = train_test_split(X_padded, Y_indices, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=X_padded.shape[1]), # can change embedding layer specifically ouput_dim to increase nuance
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'), # can always add more neural networks to increase accuracy
    # tf.keras.layers.Dropout(0.5) can add dropouts to reduce overreliance on a specific pattern
    tf.keras.layers.Dense(len(set(Y)), activation='softmax')  # Adjusted to the number of unique classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

with open('tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)


# Save the trained model
model.save("C:\\Users\\benel\\Downloads\\youtube sentiment analysis")

