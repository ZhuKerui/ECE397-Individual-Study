from django.shortcuts import render
from django.http import JsonResponse
import logging
from . import MyEngine

engine = MyEngine.MyEngine('/root/dataset/index', '/root/dataset/first_task_dataset.json')

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
        results = engine.search(keywords,fields,['title', 'abstract', 'id'], is_strict, highlight=True, limit=limit)
    else:
        results = engine.search(keywords,fields,['title', 'abstract', 'id'], is_strict, highlight=True)
    content = []
    for data in results:
        content.append({"title":data['title'], "content":data['abstract'], "id":data['id']})

    return JsonResponse(content, safe=False)

def get_paper(request):
    id_ = request.GET.get('id')
    result = engine.search(id_, 'id', engine.fields)
    data = result[0]
    content = {}
    for field in engine.fields:
        content[field] = data[field]
    return render(request, 'paper.html', content)