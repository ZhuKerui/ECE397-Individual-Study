from django.http import request
from django.shortcuts import render
from django.http import JsonResponse
import logging
from . import relation_cluster
import os

print('Start loading cluster...')
MyCluster = relation_cluster.cluster('../../../dataset/word2vec/ori_vecs', '../../../dataset/word2vec/dep_vecs')
print('Cluster loaded!')

def index(request):
    return render(request, 'display.html')

def init_ck(request):
    global MyCluster
    ck = request.GET.get('central_keyword')
    content = {}
    if ck and MyCluster.is_exist(ck):
        content['status'] = 1
        content['central_keyword'] = ck
    else:
        content['status'] = 0
    return JsonResponse(content, safe=False)

def get_topic_related(request):
    global MyCluster
    ck = request.GET.get('central_keyword')
    tr_threshold = float(request.GET.get('topic_related_threshold'))
    fc_threshold = float(request.GET.get('function_cluster_threshold'))
    tr_list = MyCluster.get_topic_related(ck, tr_threshold)
    vocab = [item[1] for item in tr_list]
    cluster_num, cluster_set, cluster_center_vecs, w2c = MyCluster.get_function_similar_cluster(vocab, fc_threshold, ck)
    content = {}
    if cluster_num is None:
        content['status'] = 0
    else:
        content['status'] = 1
        content['children'] = {}
        cluster_centers = MyCluster.find_central_word(cluster_set, cluster_center_vecs, ck)
        for w, set_ in zip(cluster_centers, cluster_set):
            content['children'][w] = list(set_)
    return JsonResponse(content, safe=False)

# def get_cluster(request):
#     return render(request, 'index.html')

# def search_paper(request):
#     keywords = request.GET.get('keywords')
#     is_strict = request.GET.get('is_strict')
#     fields = request.GET.get('field_select')
#     limit = request.GET.get('limit')
#     if limit:
#         limit = int(limit)
#         results = engine.search(keywords,fields,['title', 'abstract', 'id'], is_strict, highlight=True, limit=limit)
#     else:
#         results = engine.search(keywords,fields,['title', 'abstract', 'id'], is_strict, highlight=True)
#     content = []
#     for data in results:
#         content.append({"title":data['title'], "content":data['abstract'], "id":data['id']})

#     return JsonResponse(content, safe=False)

# def get_paper(request):
#     id_ = request.GET.get('id')
#     result = engine.search(id_, 'id', engine.fields)
#     data = result[0]
#     content = {}
#     for field in engine.fields:
#         content[field] = data[field]
#     return render(request, 'paper.html', content)