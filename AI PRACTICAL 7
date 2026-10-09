from sklearn.naive_bayes import BernoulliNB
X = [
    [1, 1, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [0, 0, 0]
]

y = [
    'Flu',
    'Flu',
    'Flu',
    'Cold',
    'Cold',
    'Cold'
]
model = BernoulliNB()

model.fit(X, y)
patient = [[1, 1, 1]]

prediction = model.predict(patient)

print("Predicted diagnosis:", prediction[0])
probabilities = model.predict_proba(patient)

for diagnosis, probability in zip(model.classes_, probabilities[0]):
    print(diagnosis, "=", round(probability, 2))
patient2 = [[0, 1, 0]]

prediction2 = model.predict(patient2)

print("Predicted diagnosis:", prediction2[0])
