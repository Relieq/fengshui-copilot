from django.shortcuts import render


def page_ask(req):
    return render(req, "ask.html")