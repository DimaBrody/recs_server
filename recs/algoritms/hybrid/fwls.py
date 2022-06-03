import random
from decimal import Decimal

from ..base.base_recs import BaseRecs
from ..collaborative.mf.mf_class import MatrixFactorizationRecs
from ..content_based.lda.lda_class import ContentBasedRecs
from ..collaborative.user_user.uu_class import UserUserRecs
import pickle

from ...models import Rating

user_len = None


class FeatureWeightedLinearStacking(BaseRecs):

    def __init__(self):
        self.cb = ContentBasedRecs()
        self.cf = MatrixFactorizationRecs()

        self.wcb1 = Decimal(0.65221204)
        self.wcb2 = Decimal(-0.14638855)
        self.wcf1 = Decimal(-0.0062952)
        self.wcf2 = Decimal(0.09139193)
        self.intercept = Decimal(0)

    @staticmethod
    def fun1():
        return Decimal(1.0)

    @staticmethod
    def fun2(user_id):
        global user_len
        if not user_len:
            user_len = Rating.objects.filter(user_id=user_id).count()

        if user_len >= 3.0:
            return Decimal(1.0)
        return Decimal(0.0)

    def set_save_path(self, save_path):
        with open(save_path + 'fwls_parameters.data', 'rb') as ub_file:
            parameters = pickle.load(ub_file)
            self.wcb1 = Decimal(parameters['cb1'])
            self.wcb2 = Decimal(parameters['cb2'])
            self.wcf1 = Decimal(parameters['cb1'])
            self.wcf2 = Decimal(parameters['cf2'])
            self.intercept = Decimal(parameters['intercept'])

    def recommend_items_by_ratings(self,
                                   user_id,
                                   num=6):

        cb_recs = self.cb.recommend_items_by_ratings(user_id, num * 5)
        cf_recs = self.cf.recommend_items(user_id, num * 5)

        return self.merge_predictions(user_id[0], cb_recs, cf_recs, num)

    def get_title(self):
        return "Гібридна"

    def recommend_items(self, user_id, num=6):
        cb_recs = self.cb.recommend_items(user_id, num * 5)
        cf_recs = self.cf.recommend_items(user_id, num * 5)

        return self.merge_predictions(user_id, cb_recs, cf_recs, num)

    def merge_predictions(self, user_id, cb_recs, cf_recs, num):

        combined_recs = dict()
        for rec in cb_recs:
            book_id = rec[0]
            pred = int(rec[1]['prediction']) - Decimal(8.5)
            combined_recs[book_id] = {'cb': pred}

        for rec in cf_recs:
            book_id = rec[0]
            pred = rec[1]
            if book_id in combined_recs.keys():
                combined_recs[book_id]['cf'] = pred
            else:
                combined_recs[book_id] = {'cf': pred}
        fwls_preds = dict()
        for key, recs in combined_recs.items():
            if 'cb' not in recs.keys():
                recs['cb'] = self.cb.predict_score(user_id, key)
            if 'cf' not in recs.keys():
                recs['cf'] = Decimal(self.cf.predict_score(user_id, key))
            pred = self.prediction(recs['cb'], recs['cf'], user_id)
            fwls_preds[key] = {'prediction': pred}
        sorted_items = sorted(fwls_preds.items(),
                              key=lambda item: -float(item[1]['prediction']))
        return sorted_items[:num]

    def predict_score(self, user_id, item_id):
        p_cb = self.cb.predict_score(user_id, item_id)
        p_cf = self.cf.predict_score(user_id, item_id)

        self.prediction(p_cb, p_cf, user_id)

    def prediction(self, p_cb, p_cf, user_id):
        p = (self.wcb1 * self.fun1() * p_cb +
             self.wcb2 * self.fun2(user_id) * p_cb +
             self.wcf1 * self.fun1() * Decimal(p_cf) +
             self.wcf2 * self.fun2(user_id) * Decimal(p_cf))
        return p + self.intercept
