# from django.apps import AppConfig
# from jobs import updater
# from django.apps import apps
#
#
# class MainConfig(AppConfig):
#     default_auto_field = 'django.db.models.BigAutoField'
#     name = 'main'
#
#     def ready(self):
#         if not apps.ready:
#             updater.start()
#         # updater.start()
