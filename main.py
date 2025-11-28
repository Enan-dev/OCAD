import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import re


try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    print("NLTK resources downloaded successfully")
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")
    exit()


try:
    df = pd.read_csv("attention_log.csv")
    print("Successfully loaded attention_log.csv")
    print(f"Dataset contains {len(df)} records")
except FileNotFoundError:
    print("Error: attention_log.csv not found in current directory")
    exit()


print("\nSample data:")
print(df.head())


print("\nMissing values:")
print(df.isnull().sum())


df = df.dropna()


def preprocess_text(text):
   
    text = str(text).lower()
    
    
    text = re.sub(r'[^a-z\s]', '', text)
    
    
    tokens = text.split()
    
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
   
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)


df['combined_features'] = df['Gaze_Status'] + " " + df['Expression'] + " " + df['Engagement'] + " " + df['Final_Status']


df['processed_features'] = df['combined_features'].apply(preprocess_text)


df['target'] = df['Attention'].apply(lambda x: 1 if x == 'attentive' else 0)

print("\nTarget distribution:")
print(df['target'].value_counts())
print(f"\nAttentive percentage: {df['target'].mean()*100:.2f}%")


sia = SentimentIntensityAnalyzer()


def get_sentiment(text):
    try:
        return sia.polarity_scores(text)
    except:
        return {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}

df['sentiment_scores'] = df['processed_features'].apply(get_sentiment)


df['sentiment_neg'] = df['sentiment_scores'].apply(lambda x: x['neg'])
df['sentiment_neu'] = df['sentiment_scores'].apply(lambda x: x['neu'])
df['sentiment_pos'] = df['sentiment_scores'].apply(lambda x: x['pos'])
df['sentiment_compound'] = df['sentiment_scores'].apply(lambda x: x['compound'])


plt.figure(figsize=(10, 6))
sns.histplot(df['sentiment_compound'], bins=20, kde=True)
plt.title('Distribution of Sentiment Compound Scores')
plt.xlabel('Sentiment Compound Score')
plt.ylabel('Frequency')
plt.savefig('sentiment_distribution.png')
plt.close()


vectorizer = TfidfVectorizer(max_features=100)
X_tfidf = vectorizer.fit_transform(df['processed_features']).toarray()


X_sentiment = df[['sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound']].values
X = np.concatenate((X_tfidf, X_sentiment), axis=1)
y = df['target'].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


feature_names = list(vectorizer.get_feature_names_out()) + ['neg', 'neu', 'pos', 'compound']
feature_importances = model.feature_importances_
sorted_idx = feature_importances.argsort()[::-1]

print("\nTop 20 Important Features:")
for i in sorted_idx[:20]:
    print(f"{feature_names[i]}: {feature_importances[i]:.4f}")


plt.figure(figsize=(12, 8))
sns.barplot(x=[feature_names[i] for i in sorted_idx[:20]], 
            y=[feature_importances[i] for i in sorted_idx[:20]])
plt.title('Top 20 Important Features for Attentiveness Prediction')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('feature_importances.png')
plt.close()


print("\nSentiment Analysis Predictions:")
test_samples = [
    "Attentive Neutral bored Neutral",        # attentive
    "Looking Right Neutral bored Distracted", # Not attentive
    "Attentive Sad confused Confused",        # Attentive (but confused)
    "Looking Away Angry distracted Distracted" # Not attentive
]

for text in test_samples:
    processed = preprocess_text(text)
    scores = sia.polarity_scores(processed)
    print(f"\nText: {text}")
    print(f"Processed: {processed}")
    print(f"Sentiment: neg={scores['neg']:.2f}, neu={scores['neu']:.2f}, pos={scores['pos']:.2f}, compound={scores['compound']:.2f}")
    
   
    tfidf_vec = vectorizer.transform([processed]).toarray()
    sentiment_vec = np.array([[scores['neg'], scores['neu'], scores['pos'], scores['compound']]])
    features = np.concatenate((tfidf_vec, sentiment_vec), axis=1)
    
    prediction = model.predict(features)
    proba = model.predict_proba(features)
    status = "Attentive" if prediction[0] == 1 else "Not attentive"
    print(f"Prediction: {status} (Confidence: {proba[0][prediction[0]]:.2f})")


import joblib
joblib.dump(model, 'attention_predictor.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
print("\nModel saved as 'attention_predictor.joblib'")

print("Vectorizer saved as 'tfidf_vectorizer.joblib'")
