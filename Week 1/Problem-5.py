import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

# Create synthetic feedback dataset
good_feedback = [
    "Great quality product!", "Very satisfied with the purchase.", "Absolutely love it!",
    "Will buy again.", "Excellent and durable.", "Fantastic value for money.",
    "Highly recommend it.", "Perfect item as described.", "Happy with the service.",
    "Loved it!", "Quick delivery and good quality.", "Impressive build quality.",
    "Good for daily use.", "Wonderful experience.", "Smooth transaction.",
    "Best product I’ve used.", "Reliable and well-made.", "Great customer support.",
    "Works as expected.", "Top-notch packaging.", "Premium feel.",
    "Just what I needed!", "Totally worth it.", "Fast shipping and nice product.",
    "Superb material.", "Exactly what I wanted.", "Met all expectations.",
    "Very pleased.", "Extremely helpful and useful.", "Top quality!",
    "Good price and fast delivery.", "Performs flawlessly.", "Convenient and effective.",
    "Super satisfied.", "Would buy again.", "Nice product.",
    "Totally satisfied.", "It’s just perfect.", "Better than expected.",
    "Great design and usability.", "Works perfectly.", "Just amazing.",
    "Five stars!", "Great shopping experience.", "Everything was perfect.",
    "Great deal.", "Nice packaging.", "Excellent performance.",
    "Great value.", "Functional and stylish."
]

bad_feedback = [
    "Terrible product.", "Not worth the money.", "Very disappointing.",
    "It broke after one use.", "Too cheap and flimsy.", "I want a refund.",
    "Do not recommend.", "Poor quality.", "Worst purchase ever.",
    "It arrived damaged.", "Very bad experience.", "Waste of money.",
    "Defective item.", "The product is useless.", "Did not work as expected.",
    "Cheaply made.", "Not as described.", "Doesn’t function properly.",
    "Extremely dissatisfied.", "Low-quality product.", "Horrible customer service.",
    "Packaging was bad.", "Complete waste.", "It stopped working quickly.",
    "Does not match description.", "No value for the price.", "Poor build quality.",
    "Very noisy.", "Overheats quickly.", "Shipping was very slow.",
    "Doesn't do the job.", "Regret buying it.", "Looks good but works bad.",
    "Came scratched.", "Not durable.", "Performance is poor.",
    "Bad finish.", "Feels very cheap.", "Not user-friendly.",
    "Not reliable.", "Wasted my time.", "I hate it.", "Very frustrating.",
    "Terribly made.", "Expected better.", "Never buying again.",
    "Pathetic experience.", "Malfunctioned on first day.",
    "Returned it immediately.", "Not happy with it."
]

# Combine into DataFrame
df = pd.DataFrame({
    "Feedback": good_feedback + bad_feedback,
    "Label": ["good"] * 50 + ["bad"] * 50
})

# 1st part - Preprocessing using TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=300, stop_words='english', lowercase=True)
X = vectorizer.fit_transform(df["Feedback"])
y = df["Label"]

# 2nd part - Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 3rd part - Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


precision = precision_score(y_test, y_pred, pos_label="good")
recall = recall_score(y_test, y_pred, pos_label="good")
f1 = f1_score(y_test, y_pred, pos_label="good")

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# 4th part - Text vectorization function
def text_preprocess_vectorize(texts, vectorizer):
    return vectorizer.transform(texts)