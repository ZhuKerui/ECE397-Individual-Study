from django.shortcuts import render
from django.http import JsonResponse
import logging
from . import MyEngine

# engine = MyEngine.MyEngine('/root/dataset/related_kw_index', '/root/dataset/related_words.json') # for docker
engine = MyEngine.MyEngine('../../dataset/related_kw_index', '../../root/dataset/related_words.json') # for docker

def index(request):
    logging.debug('debug works')
    return render(request, 'index.html')

def search_related(request):
    logging.debug('Search request received')
    keywords = request.GET.get('keywords')
    local_search = request.GET.get('local_search')
    limit = request.GET.get('limit')
    if limit:
        limit = int(limit)
        results = engine.search(keywords, local_search=local_search, limit=limit)
    else:
        results = engine.search(keywords, local_search=local_search)
    content = []
    for keyword, suggested_overlap in results.items():
        content.append({"title":keyword, "content":', '.join(suggested_overlap)})

    return JsonResponse(content, safe=False)
