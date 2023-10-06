from django.urls import path, include
from Home.views import Homepage
urlpatterns = [
    path('', Homepage)
]
