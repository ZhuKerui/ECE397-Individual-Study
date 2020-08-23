from django.shortcuts import render
from django.http import JsonResponse
import logging
from . import MyEngine

engine = MyEngine.MyEngine('/root/dataset/test_index', '/root/dataset/snapshot.json')

def index(request):
    logging.debug('debug works')
    return render(request, 'index.html')

def search_paper(request):
    keywords = request.GET.get('keywords')
    is_strict = request.GET.get('is_strict')
    fields = request.GET.get('field_select')
    limit = request.GET.get('limit')
    if limit:
        limit = int(limit)
        results = engine.search(keywords,fields,['title', 'abstract'], True, limit)
    else:
        results = engine.search(keywords,fields,['title', 'abstract'], True)
    content = []
    for data in results:
        content.append({"title":data['title'], "content":data['abstract']})
    # content.append({"title":'title', "content":'abstract'})
    return JsonResponse(content, safe=False)