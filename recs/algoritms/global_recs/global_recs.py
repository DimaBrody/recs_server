from recs.algoritms.base.base_recs import BaseRecs
from recs.data import shared_data


class GlobalRecs(BaseRecs):

    def predict_score(self, user_object, item_id):
        pass

    def recommend_items(self, user_object, num=6):
        sorted_books = [(shared_data.book_translations_uu_old[k], len(v)) for k, v in
                        sorted(shared_data.book_to_user.items(), key=lambda item: len(item[1]), reverse=True)]

        return sorted_books[:num]

    def get_title(self):
        return "Популярні"
