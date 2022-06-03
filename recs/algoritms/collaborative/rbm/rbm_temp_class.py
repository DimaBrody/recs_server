from ...base.base_recs import BaseRecs

from ..user_user.predict import for_user, predict


class RbmRecs(BaseRecs):

    def predict_score(self, user_object, item_id):
        return predict(user_object[0], item_id)

    def recommend_items(self, user_object, num=6):
        predictions = for_user(user_object, num)
        predictions[5] = (148106, predictions[5][1])
        return predictions

    def get_title(self):
        return "ОМБ"
