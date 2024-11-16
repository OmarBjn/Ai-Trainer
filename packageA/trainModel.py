import pandas as pd 
from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score # Accuracy metrics
import pickle

# Read the dataset
df = pd.read_csv('coordsBicepCurl.csv')

X = df.drop('class', axis=1) # features
y = df['class'] # target value

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

# Train Machine Learning Classification Model
pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier())
}

fit_models = {}
for algo, pipeline in pipelines.items():
    print(f"algo : {algo}")
    print(f"pipeline : {pipeline}")
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model

# 3.3 Evaluate and Serialize Model
for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    print(algo, accuracy_score(y_test.values, yhat),
          precision_score(y_test.values, yhat, average="weighted"),
          recall_score(y_test.values, yhat, average="weighted"))   

with open('coordsBicepCurl.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)