from django.http import request
from django.shortcuts import render
from django.http import JsonResponse
import logging
from . import graph_embed
from . import relation_cluster
import os

print('Start loading graph embedding...')
MyGraphEmbed = graph_embed.Graph_Embed()
MyGraphEmbed.load_nps('../../../dataset/node2vec/vecs')
print('Embedding loaded!')

print('Start loading cluster...')
MyCluster = relation_cluster.cluster('../../../dataset/word2vec/ori_vecs', '../../../dataset/word2vec/dep_vecs')
print('Cluster loaded!')

def index(request):
    return render(request, 'display.html')

def g_get_related(request):
    global MyGraphEmbed
    sk = request.GET.get('search_keyword')
    threshold = float(request.GET.get('threshold'))
    content = {}
    if sk and MyGraphEmbed.is_word_exist(sk):
        content['status'] = 1
        content['related'] = MyGraphEmbed.get_related(sk, threshold)
    else:
        content['status'] = 0
    return JsonResponse(content, safe=False)

def w_get_related(request):
    global MyCluster
    sk = request.GET.get('search_keyword')
    threshold = float(request.GET.get('threshold'))
    content = {}
    if sk and MyCluster.is_exist(sk):
        content['status'] = 1
        content['related'] = MyCluster.get_related(sk, threshold, use_topic=True, similariy=False)
    else:
        content['status'] = 0
    return JsonResponse(content, safe=False)

def d_get_related(request):
    global MyCluster
    sk = request.GET.get('search_keyword')
    threshold = float(request.GET.get('threshold'))
    content = {}
    if sk and MyCluster.is_exist(sk):
        content['status'] = 1
        content['related'] = MyCluster.get_related(sk, threshold, use_topic=False, similariy=False)
    else:
        content['status'] = 0
    return JsonResponse(content, safe=False)
