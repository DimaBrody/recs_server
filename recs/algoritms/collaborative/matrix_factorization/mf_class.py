import pickle

from recs.algoritms.base.base_recs import BaseRecs
from ....data import shared_data
import os

print(os.getcwd())
from recs.algoritms.collaborative.matrix_factorization import mf_functions

with open("./recs/data/functional/dict/" + 'user_to_book_all.json', 'rb') as f:
    user_to_book_all = pickle.load(f)


is_first = True


class MatrixFactorizationRecs(BaseRecs):
    def predict_score(self, user_id, item_id):
        return mf_functions.predict(max(user_to_book_all.keys()) + 1, item_id)

    def recommend_items(self, user_id, num=6):
        # if is_first:
        #     mf_functions.update_for_user(100000)

        predictions = [(shared_data.book_translations_uu_old[book_id], self.predict_score(100000, book_id)) for book_id
                       in
                       shared_data.book_to_user.keys()]

        predictions = sorted(predictions, key=lambda item: item[1], reverse=True)

        return predictions[:num]

    def get_title(self):
        return "Матрична Факторизація"
