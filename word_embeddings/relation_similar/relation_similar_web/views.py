from relation_similar_web.co_occur_generator import Co_Occur_Generator
from relation_similar_web.sent2kw import Sent2KW
from relation_similar_web.pair_embed import Pair_Embed
from django.http import request
from django.shortcuts import render
from django.http import JsonResponse

# co_occur = Co_Occur_Generator()
# semantic_related = Co_Occur_Generator()
sent2kw = Sent2KW()
# co_occur.load_word_tree('../../../dataset/hurry/wordtree.json')
# semantic_related.load_word_tree('../../../dataset/hurry/wordtree.json')
sent2kw.load_word_tree('../../../dataset/hurry/wordtree.json')
sent2kw.register_files('../../../dataset/hurry/reformed', '../../../dataset/hurry/relation_record.json')
sent2kw.load_relation()
# co_occur.load_word_vector('../../../dataset/hurry/vecs')
# semantic_related.load_word_vector('../../../dataset/hurry/vecs')
# print('Start loading co-occur pairs...')
# co_occur.load_pairs('../../../dataset/hurry/co_occur.csv')
# print('Done with loading co-occur pairs')
# print('Start loading semantic-related pairs...')
# semantic_related.load_pairs('../../../dataset/hurry/semantic_related.csv')
# print('Done with loading semantic-related pairs')
pair_embed = Pair_Embed()
pair_embed.load_word_tree('../../../dataset/hurry/wordtree.json')
pair_embed.load_pair_vector('../../../dataset/hurry/pair_vecs')

def index(request):
    return render(request, 'display.html')

def sent_analysis(request):
    sent = request.GET.get('sent')
    content = {}
    content['co_occur_kw'] = sent2kw.find_kw(sent)
    content['semantic_related_kw'] = sent2kw.find_semantic_related_kw(sent)
    content['status'] = True
    return JsonResponse(content, safe=False)

def get_sent_by_relation(request):
    relation = request.GET.get('relation')
    count = int(request.GET.get('count'))
    ret = sent2kw.get_sent_by_relation(relation, count)
    content = {}
    if ret is not None:
        content['status'] = True
        content['sents_kws'] = ret
    else:
        content['status'] = False
    return JsonResponse(content, safe=False)

def search_relation(request):
    relation_count = int(request.GET.get('relation_count'))
    relations = [[rel, len(item)] for rel, item in sent2kw.relation_record.items()]
    relations.sort(key=lambda x: x[1], reverse=True)
    if relation_count > len(relations):
        relation_count = len(relations)
    content = {}
    content['status'] = True
    content['relations'] = relations[0:relation_count]
    return JsonResponse(content, safe=False)

def search(request):
    # global co_occur
    # global semantic_related
    global pair_embed
    ck = request.GET.get('central_keyword')
    # cnt_threshold = int(request.GET.get('cnt_threshold'))
    # npmi_threshold = float(request.GET.get('npmi_threshold'))
    # pair_set = request.GET.get('pair_set')
    k = int(request.GET.get('k'))
    content = {}
    # if pair_set == "co_occur":
    #     co_occur.filter(min_count=cnt_threshold, min_npmi=npmi_threshold)
    #     score, clusters = co_occur.dbscan_cluster(ck, k=k)
    # else:
    #     semantic_related.filter(min_count=cnt_threshold, min_npmi=npmi_threshold)
    #     score, clusters = semantic_related.dbscan_cluster(ck, k=k)
    score, clusters = pair_embed.dbscan_cluster(ck, k=k)
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