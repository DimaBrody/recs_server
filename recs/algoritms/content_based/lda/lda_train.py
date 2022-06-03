import os
import sqlite3
import tqdm
from scipy.sparse import coo_matrix

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'diploma.settings')

import django
from datetime import datetime

import logging
import numpy as np

import pyLDAvis
import pyLDAvis.gensim

import operator
import math

from diploma import settings

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from gensim import corpora, models, similarities
from ....models import BookCB, LdaSimilarity
from django.forms.models import model_to_dict


def dot_product(v1, v2):
    dp = sum(map(operator.mul, v1, v2))
    return dp


def vector_cos(v1, v2):
    prod = dot_product(v1, v2)
    sqrt1 = math.sqrt(dot_product(v1, v1))
    sqrt2 = math.sqrt(dot_product(v2, v2))
    return prod / (sqrt1 * sqrt2)


def cosine_similarity(ldas):
    size = ldas.shape[0]
    similarity_matrix = np.zeros((size, size))

    for i in range(ldas.shape[0]):

        for j in range(ldas.shape[0]):
            similarity_matrix[i, j] = vector_cos(ldas[i,], ldas[j,])

    return similarity_matrix


def load_data():
    docs = list(BookCB.objects.all())
    data = ["{}, {}, {}".format(d.title, d.genres, d.description) for d in docs]

    if len(data) == 0:
        print("No descriptions were found, run populate_sample_of_descriptions")
    return data, docs


class LdaModel(object):

    def __init__(self, min_sim=0.15):
        self.dirname, self.filename = os.path.split(os.path.abspath(__file__))
        self.min_sim = min_sim

    def train(self, data=None, docs=None):

        if data is None:
            data, docs = load_data()

        NUM_TOPICS = 10

        self.lda_path = "./recs/algoritms/content_based/lda/static/"

        print(os.path.isdir(self.lda_path))
        # if not os.path.exists(self.lda_path):
        #     os.makedirs(self.lda_path)

        self.build_lda_model(data, docs, NUM_TOPICS)

    @staticmethod
    def tokenize(self, data):
        tokenizer = RegexpTokenizer(r'\w+')

        return [tokenizer.tokenize(d) for d in data]

    def build_lda_model(self, data, docs, n_topics=5):

        texts = []
        tokenizer = RegexpTokenizer(r'\w+')
        for d in data:
            raw = d.lower()

            tokens = tokenizer.tokenize(raw)

            stopped_tokens = self.remove_stopwords(tokens)

            stemmed_tokens = stopped_tokens
            # stemmer = PorterStemmer()
            # stemmed_tokens = [stemmer.stem(token) for token in stopped_tokens]

            texts.append(stemmed_tokens)

        dictionary = corpora.Dictionary(texts)

        corpus = [dictionary.doc2bow(text) for text in texts]

        lda_model = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary,
                                             num_topics=n_topics)

        index = similarities.MatrixSimilarity(corpus)

        self.save_lda_model(lda_model, corpus, dictionary, index)
        self.save_similarities(index, docs)

        return dictionary, texts, lda_model

    def save_lda_model(self, lda_model, corpus, dictionary, index):

        index.save(self.lda_path + 'index.lda')
        print(os.getcwd())
        pyLDAvis.save_json(pyLDAvis.gensim.prepare(lda_model, corpus, dictionary), self.lda_path + 'lda.json')
        print(lda_model.print_topics())
        lda_model.save(self.lda_path + 'model.lda')

        dictionary.save(self.lda_path + 'dict.lda')
        corpora.MmCorpus.serialize(self.lda_path + 'corpus.mm', corpus)

    @staticmethod
    def remove_stopwords(tokenized_data):

        en_stop = get_stop_words('en')
        en_stop.append('s')

        stopped_tokens = [token for token in tokenized_data if token not in en_stop]
        return stopped_tokens

    def save_similarities(self, index, docs, created=datetime.now()):
        self.save_similarities_with_django(index, docs, created)

    def save_similarities_with_django(self, index, docs, created=datetime.now()):
        start_time = datetime.now()
        print(f'truncating table in {datetime.now() - start_time} seconds')

        no_saved = 0
        start_time = datetime.now()
        coo = coo_matrix(index)
        csr = coo.tocsr()

        print(f'instantiation of coo_matrix in {datetime.now() - start_time} seconds')

        conn = self.get_conn()
        cur = conn.cursor()

        try:
            cur.execute('truncate table lda_similarity')
        except sqlite3.OperationalError:
            print("Error to truncate")
            pass

        nonzero_count = coo.count_nonzero()
        print(f'{nonzero_count} - всього моделей')

        xs, ys = coo.nonzero()
        idx = 0
        for x, y in zip(xs, ys):

            if x == y:
                continue

            sim = float(csr[x, y])
            x_id = str(docs[x].id)
            y_id = str(docs[y].id)

            idx += 1

            if idx % 1000000 == 0:
                print("процес: %.2f" % ((idx / nonzero_count) * 100) + "%")

            if sim < self.min_sim:
                continue

            similarities_inner = []
            if x_id is not None and y_id is not None and sim is not None:
                # similarities_inner.append(LdaSimilarity(source=x_id, target=y_id, similarity=sim))
                no_saved += 1

            # LdaSimilarity.objects.bulk_create(similarities_inner)

        print('{} LDA моделей збережено для min_sim > {}, за {}'.format(no_saved, self.min_sim,
                                                                        datetime.now() - start_time))

    @staticmethod
    def get_conn():
        dbName = settings.DATABASES['default']['NAME']
        conn = sqlite3.connect(dbName)
        return conn


def main():
    print("Calculating lda model...")

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    data, docs = load_data()

    lda = LdaModel()
    lda.train(data, docs)
