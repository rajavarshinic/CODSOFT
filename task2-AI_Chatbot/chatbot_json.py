#import necessary libraries
import numpy as np
import json
import random
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import nltk
nltk.download('wordnet')

# Load the intents from the JSON file
with open('intents.json', 'r') as file:
    intents = json.load(file)['intents']
    
#Data Preparation and exploration phase-2


'''
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df['tag'].unique())
df_transposed = df.transpose()
print(df_transposed)
merged_data = pd.merge(df, df, on='tag') 
print(merged_data)
grouped_data = df.groupby('tag')
aggregated_data = grouped_data['responses'].apply(len)
print(aggregated_data)
import seaborn as sns
import matplotlib.pyplot as plt 

# Univariate analysis - Histogram 
sns.histplot(df['tag'], bins=20)

plt.show() 


# Bivariate analysis - Scatter plot 
df['num_patterns'] = df['patterns'].apply(len)
df['num_responses'] = df['responses'].apply(len)
sns.scatterplot(x='num_patterns', y='num_responses', data=df)
plt.show()

# Multivariate analysis - Pair plot 
sns.pairplot(df) 
plt.show()
df['num_patterns'] = df['patterns'].apply(len)
user_profiles = df.groupby('tag').agg({'num_patterns': 'mean'})
print(user_profiles)

'''

# Initialize the lemmatizer and tokenizer
lemmatizer = WordNetLemmatizer()
tokenizer = word_tokenize
# Create a list of all words in the intents, and a list of all intents
words = []
classes = []
documents = []


for intent in intents:
    for pattern in intent['patterns']:
        # Tokenize and lemmatize each word in the pattern
        words_in_pattern = tokenizer(pattern.lower())
        words_in_pattern = [lemmatizer.lemmatize(word) for word in words_in_pattern]
        # Add the words to the list of all words
        words.extend(words_in_pattern)
        # Add the pattern and intent to the list of all documents
        documents.append((words_in_pattern, intent['tag']))
        # Add the intent to the list of all intents
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
# Remove duplicates and sort the words and classes
words = sorted(list(set(words)))
classes = sorted(classes)
# Create training data as a bag of words
training_data = []

for document in documents:
    bag = []
    # Create a bag of words for each document
    for word in words:
        bag.append(1) if word in document[0] else bag.append(0)
    # Append the bag of words and the intent tag to the training data
    output_row = [0] * len(classes)
    output_row[classes.index(document[1])] = 1
    training_data.append([bag, output_row])
# Shuffle the training data and split it into input and output lists
random.shuffle(training_data)
training_data = np.array(training_data, dtype=object)
train_x = list(training_data[:, 0])
train_y = list(training_data[:, 1])
# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
])
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5)
# Define a function to process user input and generate a response
def get_response(user_input):
    # Tokenize and lemmatize the user input
    words_in_input = tokenizer(user_input.lower())
    words_in_input = [lemmatizer.lemmatize(word) for word in words_in_input]
    
    # Create a bag of words for the user input
    bag = [0] * len(words)
    for word in words_in_input:
        for i, w in enumerate(words):
            if w == word:
                bag[i] = 1
    
    # Predict the intent of the user input using the trained model
    results = model.predict(np.array([bag]), verbose=0)[0]
    # Get the index of the highest probability result
    index = np.argmax(results)
    # Get the corresponding intent tag
    tag = classes[index]
    
    # If the probability of the predicted intent is below a certain threshold, return a default response
    if results[index] < 0.5:
        return "I'm sorry, I don't understand. Can you please rephrase?"
    
    # Get a random response from the intent
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
    
    return response

# Main loop to get user input and generate responses
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
            break
    response = get_response(user_input)
    print("RaBot:", response)


# Import necessary libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Convert the model's prediction probabilities to class labels
y_pred = np.argmax(model.predict(np.array(train_x)), axis=1)

# Convert the one-hot encoded class labels to their integer format
y_true = np.argmax(np.array(train_y), axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy}')

# Calculate precision
precision = precision_score(y_true, y_pred, average='weighted')
print(f'Precision: {precision}')

# Calculate recall
recall = recall_score(y_true, y_pred, average='weighted')
print(f'Recall: {recall}')

# Calculate F1 score
f1 = f1_score(y_true, y_pred, average='weighted')
print(f'F1 Score: {f1}')

