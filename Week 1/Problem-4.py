import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Create Synthetic Dataset
positive_reviews = [
    "Amazing movie!", "Loved the acting.", "What a great story!",
    "Brilliant direction!", "Enjoyed every moment.", "Excellent cast.",
    "Highly recommended.", "Great visuals and plot.", "Superb film!",
    "Incredible experience.", "Touching and beautiful.", "Great soundtrack!",
    "A masterpiece.", "Wonderful film!", "Top-notch acting.",
    "Really funny and heartwarming.", "Such a feel-good movie.",
    "Loved the characters.", "Five stars!", "Worth watching.",
    "Emotionally moving.", "Perfect film night choice.", "Hilarious scenes.",
    "Well-written dialogues.", "Fascinating and deep.",
    "Outstanding work!", "Terrific performance.", "Would watch again.",
    "A must-see!", "Loved every second.", "Positive vibes all around.",
    "A thrilling ride!", "Magical storytelling.", "Smart and engaging.",
    "Fantastic job!", "Oscar-worthy!", "Very impressive.",
    "Pure joy to watch.", "The movie was amazing.", "Fun and inspiring.",
    "Absolutely loved it.", "Stunning cinematography.",
    "It was brilliant.", "Best movie of the year.", "Laughed a lot.",
    "Inspirational story.", "Just awesome!", "Classy and emotional.",
    "Truly remarkable.", "Feel-good and warm."
]

negative_reviews = [
    "Terrible movie.", "Hated it.", "Very boring.", "Waste of time.",
    "Disappointing.", "Awful plot.", "Bad acting.", "Worst movie ever.",
    "Painful to watch.", "So dull.", "Poor direction.", "I regret watching it.",
    "Zero stars.", "Horrible experience.", "Nothing good about it.",
    "Unwatchable.", "Stupid storyline.", "Extremely slow.",
    "Predictable and lame.", "Terrible visuals.", "Unbelievable acting.",
    "Low quality.", "So much cringe.", "Unrealistic characters.",
    "Not worth your time.", "Weak dialogues.", "Total mess.",
    "Such a flop.", "Avoid at all costs.", "Too long and boring.",
    "Just awful.", "Embarrassing.", "Very poor effort.",
    "Really bad movie.", "I walked out.", "What a disaster.",
    "Forgettable.", "Absolutely terrible.", "Snoozefest.",
    "Didn't like it at all.", "No story at all.", "Full of clichés.",
    "Terrible pacing.", "Dumb and pointless.", "Mediocre at best.",
    "Uninspired.", "Feels unfinished.", "Disaster of a film.",
    "So bad it’s funny.", "Zero entertainment."
]

# Combine into a DataFrame
reviews = pd.DataFrame({
    "Review": positive_reviews + negative_reviews,
    "Sentiment": ["positive"] * 50 + ["negative"] * 50
})

# 1st part - Tokenization
vectorizer = CountVectorizer(stop_words='english', max_features=500)
X = vectorizer.fit_transform(reviews["Review"])
y = reviews["Sentiment"]

# 2nd part - Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3rd part - Train the Model
model = MultinomialNB()
model.fit(X_train, y_train)

# 3rd part - Predict and Print Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test set:", accuracy)

# 4th part - Sentiment Prediction Function
def predict_review_sentiment(model, vectorizer, review):
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)
    return prediction[0]