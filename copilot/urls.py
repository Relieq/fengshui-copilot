from django.urls import path

from .views import api_ask, page_ask

urlpatterns = [
    path("api/ask", api_ask, name="api_ask"),
    path("ask", page_ask, name="page_ask"),
]