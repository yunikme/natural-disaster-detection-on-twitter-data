from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, accuracy_score

#Step 0: Load the dataset to inspect its structure
import pandas as pd
file_path = 'train.csv'
data = pd.read_csv(file_path)

#Step 1: Preprocess the data 
#Fill missing keyword and location with empty strings
data['keyword'] = data ['keyword'].fillna('')
data ['location'] = data ['location'].fillna ('')

#combine text, keyword, and location as features
data['combined_features'] = data['keyword']+' '+ data['location']+' '+ data['text']

#Step 2: Split the data into training and testing sets 
X = data ['combined_features']
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify=y )
print(X_train)
print("ini X:", X)

#Step 3: Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range = (1,2), stop_words= 'english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print("X_train_tfidf:", X_train_tfidf) 
print("X_test_tfidf:", X_test_tfidf)

#Step 4: Train a supervised model (Logistic Regression)
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf,y_train)

#step 5: make predictions and evaluate the model
y_pred = model.predict(X_test_tfidf)

#Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print (accuracy, classification_rep)
