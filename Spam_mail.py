import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
df=pd.read_csv("/content/mail_data.csv")
df = df.where((pd.notnull(df)),"")
df.head()
df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})
x = df['Message']
y = df['Category']
print(x)
print(y)
print(x.shape)
print(y.shape)
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
x_features = vectorizer.fit_transform(x)
print(x_features.shape)
x_train, x_test, y_train, y_test = train_test_split(x_features, y, test_size=0.2, random_state=42)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
model=SVC()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
