import pickle

averages = {}
deviations = {}
neighbors = {}


def setup_data():
    global averages
    global deviations
    global neighbors

    averages.clear()
    deviations.clear()
    neighbors.clear()

    with open('./recs/algoritms/collaborative/user_user/uu_averages.json', 'rb') as f:
        averages = pickle.load(f)

    with open('./recs/algoritms/collaborative/user_user/uu_deviations.json', 'rb') as f:
        deviations = pickle.load(f)

    with open('./recs/algoritms/collaborative/user_user/uu_neighbors.json', 'rb') as f:
        neighbors = pickle.load(f)
