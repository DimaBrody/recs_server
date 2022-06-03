from decimal import Decimal

from django.db.models import Q

# from analytics.models import Rating
from django.forms.models import model_to_dict
from ....models import LdaSimilarity, BookCB, Book, Rating
from ...base.base_recs import BaseRecs

lda_path = './lda/'


class ContentBasedRecs(BaseRecs):

    def __init__(self, min_sim=0.1):

        self.min_sim = min_sim
        self.max_candidates = 100

    def recommend_items(self, user_id, num=10):
        active_user_items = [model_to_dict(rating) for rating in
                             Rating.objects.filter(user_id=user_id).order_by('-rating')[:100]]
        return self.recommend_items_by_ratings(user_id, active_user_items, num)

    def get_title(self):
        return "Змістовно базована"

    @staticmethod
    def seeded_rec(content_ids, take=6):
        data = LdaSimilarity.objects.filter(source__in=content_ids) \
                   .order_by('-similarity') \
                   .values('target', 'similarity')[:take]
        return list(data)

    def recommend_items_by_ratings(self,
                                   user_id,
                                   active_user_items,
                                   num=6):
        if len(active_user_items) == 0:
            return {}

        book_ids = {int(book['book_id']): book['rating'] for book in active_user_items}

        user_mean = sum(book_ids.values()) / len(book_ids)

        sims = LdaSimilarity.objects.filter(Q(source__in=book_ids.keys())
                                            & ~Q(target__in=book_ids.keys())
                                            & Q(similarity__gt=self.min_sim))


        sims = sims.order_by('-similarity')[:self.max_candidates]
        recs = dict()
        targets = set(s.target for s in sims if not s.target == '')
        for target in targets:

            pre = 0
            sim_sum = 0

            rated_items = [i for i in sims if i.target == target]

            if len(rated_items) > 0:

                for sim_item in rated_items:
                    r = Decimal(book_ids[int(sim_item.source)] - user_mean)
                    pre += sim_item.similarity * r
                    sim_sum += sim_item.similarity

                if sim_sum > 0:
                    recs[target] = {'prediction': Decimal(user_mean) + pre / sim_sum,
                                    'sim_items': [r.source for r in rated_items]}

        # recs.pop('85456')
        sorted_items = sorted(recs.items(), key=lambda item: -float(item[1]['prediction']))[:num]
        return sorted_items[:num]

    def predict_score(self, user_id, item_id):
        user_items = (Rating.objects.filter(user_id=user_id)
                          .exclude(book_id=item_id)
                          .order_by('-rating').values()[:100])

        book_ids = {int(book['book_id']): book['rating'] for book in user_items}

        user_mean = sum(book_ids.values()) / len(book_ids)

        sims = LdaSimilarity.objects.filter(Q(source__in=book_ids.keys())
                                            & Q(target=item_id)
                                            & Q(similarity__gt=self.min_sim)).order_by('-similarity')

        pre = 0
        sim_sum = 0
        prediction = Decimal(0.0)

        if len(sims) > 0:
            for sim_item in sims:
                r = Decimal(book_ids[int(sim_item.source)] - user_mean)
                pre += sim_item.similarity * r
                sim_sum += sim_item.similarity

            prediction = Decimal(user_mean) + pre / sim_sum
        return prediction


def get_book_ids(sorted_sims):
    ids = [s[0] for s in sorted_sims]
    books = list(BookCB.objects.filter(lda_vector__in=ids))
    m_info = {str(book.lda_vector): (book.id, book.title, sorted_sims) for book in books}

    return [{"target": m_info[str(id[0])][0],
             "title": m_info[str(id[0])][1],
             "sim": str(id[1])} for id in sorted_sims]
