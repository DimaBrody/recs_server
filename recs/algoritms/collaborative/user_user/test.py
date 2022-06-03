import pickle
import numpy as np
from predict import predict

start_path = '../../../data/functional/dict/'

with open('uu_averages.json', 'rb') as f:
    averages = pickle.load(f)

with open('uu_deviations.json', 'rb') as f:
    deviations = pickle.load(f)

with open('uu_neighbors.json', 'rb') as f:
    neighbors = pickle.load(f)

with open(start_path + 'user_to_book.json', 'rb') as f:
    user_to_book = pickle.load(f)

with open(start_path + 'book_to_user.json', 'rb') as f:
    book_to_user = pickle.load(f)

with open(start_path + 'user_book_to_rating.json', 'rb') as f:
    user_book_to_rating = pickle.load(f)

with open(start_path + 'user_book_to_rating_test.json', 'rb') as f:
    user_book_to_rating_test = pickle.load(f)

train_predictions = []
train_targets = []

for (i, b), target in user_book_to_rating.items():
    if i in user_to_book.keys() and b in book_to_user.keys():
        prediction = predict(i, b, neighbors, deviations, averages)

        train_predictions.append(prediction)
        train_targets.append(target)

print("Середнє значення: ", np.mean(train_predictions))

test_predictions = []
test_targets = []

for (i, b), target in user_book_to_rating_test.items():
    if i in user_to_book.keys() and b in book_to_user.keys():
        prediction = predict(i, b, neighbors, deviations, averages)

        test_predictions.append(prediction)
        test_targets.append(target)


def mse(p, t):
    p = np.array(p)
    t = np.array(t)

    print("p:", p)
    print("t: ", t)

    return np.sum((p - t) ** 2) / len(p)



print('СКП для сету тренувань:', mse(train_predictions, train_targets))
print('СКП для тестового сету:', mse(test_predictions, test_targets))
