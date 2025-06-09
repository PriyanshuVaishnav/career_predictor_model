import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

np.random.seed(42)
random.seed(42)

num_students = 200

marks_math = np.clip(np.random.normal(70, 15, num_students), 0, 100).astype(int)
marks_english = np.clip(np.random.normal(65, 20, num_students), 0, 100).astype(int)
marks_science = np.clip(np.random.normal(68, 18, num_students), 0, 100).astype(int)

interests_tech = np.random.binomial(1, 0.5, num_students)
interests_art = np.random.binomial(1, 0.4, num_students)
interests_sports = np.random.binomial(1, 0.3, num_students)
interests_communication = np.random.binomial(1, 0.4, num_students)

aptitude_score = ((marks_math + marks_english + marks_science) / 3) + np.random.normal(0, 10, num_students)
aptitude_score = np.clip(aptitude_score, 0, 100).astype(int)

openness = np.clip(np.random.normal(60, 20, num_students), 0, 100).astype(int)
conscientiousness = np.clip(np.random.normal(65, 15, num_students), 0, 100).astype(int)
extroversion = np.clip(np.random.normal(55, 25, num_students), 0, 100).astype(int)
agreeableness = np.clip(np.random.normal(70, 10, num_students), 0, 100).astype(int)
neuroticism = np.clip(np.random.normal(40, 20, num_students), 0, 100).astype(int)

career = []
for i in range(num_students):
    if interests_tech[i] == 1 and marks_math[i] > 65 and aptitude_score[i] > 60:
        career.append('ML Engineer')
    elif interests_art[i] == 1 and extroversion[i] > 50:
        career.append('Graphic Designer')
    elif interests_communication[i] == 1 and conscientiousness[i] > 60:
        career.append('Lawyer')
    elif interests_tech[i] == 0 and interests_communication[i] == 1 and marks_english[i] > 70:
        career.append('Civil Services')
    elif interests_sports[i] == 1 and extroversion[i] > 60:
        career.append('Sports Person')
    else:
        career.append('Doctor')

df = pd.DataFrame({
    'marks_math': marks_math,
    'marks_english': marks_english,
    'marks_science': marks_science,
    'interest_tech': interests_tech,
    'interest_art': interests_art,
    'interest_sports': interests_sports,
    'interest_communication': interests_communication,
    'aptitude_score': aptitude_score,
    'openness': openness,
    'conscientiousness': conscientiousness,
    'extroversion': extroversion,
    'agreeableness': agreeableness,
    'neuroticism': neuroticism,
    'career': career
})

# Save dataset (optional)
df.to_csv('synthetic_student_career_dataset.csv', index=False)

X = df.drop('career', axis=1)
y = df['career']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

model = DecisionTreeClassifier(random_state=42, max_depth=5)
model.fit(X_train, y_train)

joblib.dump(model, 'career_predictor_model.joblib')
joblib.dump(le, 'label_encoder.joblib')

print("Model and label encoder saved!")
