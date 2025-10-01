from django.urls import path

from .views import api_ask, page_ask
from .views.api import api_ask_stream

urlpatterns = [
    path("api/ask", api_ask, name="api_ask"),
    path("api/ask/stream", api_ask_stream, name="api_ask_stream"),
    path("ask", page_ask, name="page_ask"),
]