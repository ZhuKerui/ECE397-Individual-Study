from django.shortcuts import render
from django.http import JsonResponse

def index(request):
    return render(request, 'index.html')

def search_paper(request):
    keywords = request.GET.get('keywords')
    is_strict = request.GET.get('is_strict')
    fields = request.GET.get('field_select')
    content = []
    content.append({"title":keywords, "content":"content_1"})
    content.append({"title":is_strict, "content":"content_2"})
    content.append({"title":fields, "content":"content_2"})
    return JsonResponse(content, safe=False)