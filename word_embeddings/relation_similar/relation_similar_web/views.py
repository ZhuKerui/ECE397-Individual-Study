from relation_similar_web.co_occur_generator import Co_Occur_Generator
from django.http import request
from django.shortcuts import render
from django.http import JsonResponse
import logging

# co_occur = Co_Occur_Generator()
semantic_related = Co_Occur_Generator()
# co_occur.load_word_tree('../../../dataset/relation_similar/wordtree.json')
semantic_related.load_word_tree('../../../dataset/relation_similar/wordtree.json')
# co_occur.load_word_vector('../../../dataset/relation_similar/big_vecs')
semantic_related.load_word_vector('../../../dataset/relation_similar/big_vecs')
print('Start loading co-occur pairs...')
# co_occur.load_pairs('../../../dataset/relation_similar/big_co_occur.csv')
print('Done with loading co-occur pairs')
print('Start loading semantic-related pairs...')
semantic_related.load_pairs('../../../dataset/relation_similar/big_semantic_related.csv')
print('Done with loading semantic-related pairs')

def index(request):
    return render(request, 'display.html')

def search(request):
    # global co_occur/
    global semantic_related
    ck = request.GET.get('central_keyword')
    cnt_threshold = int(request.GET.get('cnt_threshold'))
    npmi_threshold = float(request.GET.get('npmi_threshold'))
    pair_set = request.GET.get('pair_set')
    eps = float(request.GET.get('eps'))
    min_samples = int(request.GET.get('min_samples'))
    content = {}
    # if pair_set == "co_occur":
    #     co_occur.filter(min_count=cnt_threshold, min_npmi=npmi_threshold)
    #     clusters = co_occur.dbscan_cluster(ck, eps=eps, min_samples=min_samples)
    # else:
    semantic_related.filter(min_count=cnt_threshold, min_npmi=npmi_threshold)
    score, clusters = semantic_related.dbscan_cluster(ck, eps=eps, min_samples=min_samples)
    if clusters is None:
        content['status'] = 0
    else:
        content['status'] = 1
        content['clusters'] = [list(cluster) for cluster in clusters]
        content['score'] = score
    return JsonResponse(content, safe=False)

def cal_similarity(request):
    kw1 = request.GET.get('keyword_1')
    kw2 = request.GET.get('keyword_2')
    similarity = semantic_related.get_similarity(kw1, kw2)
    content = {}
    content['status'] = similarity is not None
    if content['status']:
        content['distance'] = 1 - similarity
    return JsonResponse(content, safe=False)