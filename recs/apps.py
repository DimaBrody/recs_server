from django.apps import AppConfig
from .algoritms.collaborative.user_user import uu_data
from .data import shared_data
from .algoritms.collaborative.rbm import rbm_simple


class RecsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'recs'

    def ready(self):
        print("Loading data...")
        uu_data.setup_data()
        shared_data.setup_data()