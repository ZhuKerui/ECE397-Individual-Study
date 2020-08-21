from django.shortcuts import render

def index(request):
    # content = {}
    # content['hello'] = 'Hello World'
    return render(request, 'index.html')