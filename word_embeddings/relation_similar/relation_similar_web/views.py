# import sys 
# sys.path.append("..") 
# from co_occur_generator import Co_Occur_Generator
# from sent2kw import Sent2KW
# from pair_embed import Pair_Embed
# from django.http import request
# from django.shortcuts import render
# from django.http import JsonResponse

# co_occur = Co_Occur_Generator()
# semantic_related = Co_Occur_Generator()
# sent2kw = Sent2KW()
# co_occur.load_word_tree('../../../dataset/hurry/wordtree.json')
# semantic_related.load_word_tree('../../../dataset/hurry/wordtree.json')
# sent2kw.load_word_tree('../../../dataset/hurry/wordtree.json')
# sent2kw.register_files('../../../dataset/hurry/reformed', '../../../dataset/hurry/relation_record.json')
# sent2kw.load_relation()
# co_occur.load_word_vector('../../../dataset/hurry/vecs')
# semantic_related.load_word_vector('../../../dataset/hurry/vecs')
# # print('Start loading co-occur pairs...')
# co_occur.load_pairs('../../../dataset/hurry/co_occur.csv')
# # print('Done with loading co-occur pairs')
# # print('Start loading semantic-related pairs...')
# semantic_related.load_pairs('../../../dataset/hurry/semantic_related.csv')
# # print('Done with loading semantic-related pairs')
# # pair_embed = Pair_Embed()
# # pair_embed.load_word_tree('../../../dataset/hurry/wordtree.json')
# # pair_embed.load_word_vector('../../../dataset/hurry/big_pair_vecs')

# def index(request):
#     return render(request, 'display.html')

# def sent_analysis(request):
#     sent = request.GET.get('sent')
#     content = {}
#     content['co_occur_kw'] = sent2kw.find_kw(sent)
#     content['semantic_related_kw'] = sent2kw.find_semantic_related_kw(sent)
#     content['status'] = True
#     return JsonResponse(content, safe=False)

# def get_sent_by_relation(request):
#     relation = request.GET.get('relation')
#     count = int(request.GET.get('count'))
#     ret = sent2kw.get_sent_by_relation(relation, count)
#     content = {}
#     if ret is not None:
#         content['status'] = True
#         content['sents_kws'] = ret
#     else:
#         content['status'] = False
#     return JsonResponse(content, safe=False)

# def search_relation(request):
#     relation_count = int(request.GET.get('relation_count'))
#     relations = [[rel, len(item)] for rel, item in sent2kw.relation_record.items()]
#     relations.sort(key=lambda x: x[1], reverse=True)
#     if relation_count > len(relations):
#         relation_count = len(relations)
#     content = {}
#     content['status'] = True
#     content['relations'] = relations[0:relation_count]
#     return JsonResponse(content, safe=False)

# def search(request):
#     global co_occur
#     global semantic_related
#     # global pair_embed
#     ck = request.GET.get('central_keyword')
#     cnt_threshold = int(request.GET.get('cnt_threshold'))
#     npmi_threshold = float(request.GET.get('npmi_threshold'))
#     pair_set = request.GET.get('pair_set')
#     k = int(request.GET.get('k'))
#     content = {}
#     if pair_set == "co_occur":
#         co_occur.filter(min_count=cnt_threshold, min_npmi=npmi_threshold)
#         score, clusters = co_occur.dbscan_cluster(ck, k=k)
#     else:
#         semantic_related.filter(min_count=cnt_threshold, min_npmi=npmi_threshold)
#         score, clusters = semantic_related.dbscan_cluster(ck, k=k)
#     # score, clusters = pair_embed.dbscan_cluster(ck, k=k)
#     if clusters is None:
#         content['status'] = 0
#     else:
#         content['status'] = 1
#         content['clusters'] = [list(cluster) for cluster in clusters]
#         content['score'] = score
#     return JsonResponse(content, safe=False)

# def find_similar_pairs(request):
#     global pair_embed
#     central_keyword = request.GET.get('central_keyword')
#     similar_pair_num = request.GET.get('similar_pair_num')
#     related_keyword = request.GET.get('related_keyword')
#     ret = pair_embed.find_similar_pairs(central_keyword, related_keyword, int(similar_pair_num))
#     content = {}
#     if ret is not None:
#         content['status'] = True
#         content['similar_pairs'] = ret
#     else:
#         content['status'] = False
#     return JsonResponse(content, safe=False)

# def cal_similarity(request):
#     kw1 = request.GET.get('keyword_1')
#     kw2 = request.GET.get('keyword_2')
#     similarity = semantic_related.get_similarity(kw1, kw2)
#     content = {}
#     content['status'] = similarity is not None
#     if content['status']:
#         content['distance'] = 1 - similarity
#     return JsonResponse(content, safe=False)

import sys
sys.path.append("..")
from django.http import request
from django.shortcuts import render
from django.http import JsonResponse

from co_occur_generator import Co_Occur_Generator, dbscan_cluster
from pair_embed import Pair_Embed
from my_keywords import Keyword_Vocab, Vocab_Base
from dep_generator import Dep_Based_Embed_Generator
from pair_generator import Pair_Generator

dep_key_vocab_file = '../../../dataset/test_corpus/big_key'
dep_pair_vocab_file = '../../../dataset/test_corpus/big_pair_key'
dep_key_embed_file = '../../../dataset/test_corpus/big_dep_emb.txt'
dep_pair_embed_file = '../../../dataset/test_corpus/big_dep_pair_emb.txt'
pair2vec_key_vocab_file = '../../../dataset/corpus/big_key'
pair2vec_ctx_vocab_file = '../../../dataset/corpus/big_ctx'
pair2vec_model_file = '../../../dataset/outputs/pair2vec/test_model.pt'
co_occur_pair_file = '../../../dataset/test_corpus/big_dep_pair_train.csv'

dep_key_vocab = Keyword_Vocab()
pair2vec_key_vocab = Keyword_Vocab()
pair2vec_ctx_vocab = Vocab_Base()
print(1)

dep_key_vocab.load_vocab(dep_key_vocab_file)
dep_key_vocab.load_vector(dep_key_vocab_file)
# dep_key_vocab.read_embedding_file(dep_key_embed_file)
# dep_key_vocab.save_vector(dep_key_vocab_file)

pair2vec_key_vocab.load_vocab(pair2vec_key_vocab_file)
pair2vec_ctx_vocab.load_vocab(pair2vec_ctx_vocab_file)
print(2)

dg = Dep_Based_Embed_Generator(dep_key_vocab)

pe = Pair_Embed(dep_key_vocab)
pe.load_vocab(dep_pair_vocab_file)
pe.load_vector(dep_pair_vocab_file)
# pe.read_embedding_file(dep_pair_embed_file)
# pe.save_vector(dep_pair_vocab_file)
print(3)

pg = Pair_Generator(pair2vec_key_vocab, pair2vec_ctx_vocab)
pg.load_inference_model(pair2vec_model_file)
print(4)

cog = Co_Occur_Generator(dep_key_vocab)
cog.load_pairs(co_occur_pair_file)
print(5)

def index(request):
    return render(request, 'three_methods.html')

def search(request):
    global cog
    global dg
    global pe
    global pg
    ck = request.GET.get('central_keyword')
    # cnt_threshold = int(request.GET.get('cnt_threshold'))
    # npmi_threshold = float(request.GET.get('npmi_threshold'))
    cnt_threshold = 5
    npmi_threshold = 0.15
    # pair_set = request.GET.get('pair_set')
    k = int(request.GET.get('k'))
    related_kws = cog.get_related(ck, cnt_threshold, npmi_threshold)
    related_pairs = ['%s__%s' % (ck, kw) for kw in related_kws]
    related_key_pairs = [(ck, kw) for kw in related_kws]
    dg_score, dg_clusters = dbscan_cluster(dg.get_vectors(related_kws), related_kws, k=k)
    pe_score, pe_clusters = dbscan_cluster(pe.get_vectors(related_pairs), related_kws, k=k)
    vecs = pg.get_vectors(related_key_pairs)
    pg_score, pg_clusters = dbscan_cluster(vecs, related_kws, k=k)
    # print(vecs.shape)
    # print(len(related_kws))
    # print(type(vecs))
    content = {}
    # if dg_clusters is None or pe_clusters is None or pg_clusters is None:
    if dg_clusters is None or pe_clusters is None:
        content['status'] = 0
    else:
        content['status'] = 1
        content['dg_clusters'] = [list(cluster) for cluster in dg_clusters]
        content['dg_score'] = dg_score
        content['pe_clusters'] = [list(cluster) for cluster in pe_clusters]
        content['pe_score'] = pe_score
        content['pg_clusters'] = [list(cluster) for cluster in pg_clusters]
        content['pg_score'] = str(pg_score)
    return JsonResponse(content, safe=False)

# def find_similar_pairs(request):
#     global pair_embed
#     central_keyword = request.GET.get('central_keyword')
#     similar_pair_num = request.GET.get('similar_pair_num')
#     related_keyword = request.GET.get('related_keyword')
#     ret = pair_embed.find_similar_pairs(central_keyword, related_keyword, int(similar_pair_num))
#     content = {}
#     if ret is not None:
#         content['status'] = True
#         content['similar_pairs'] = ret
#     else:
#         content['status'] = False
#     return JsonResponse(content, safe=False)

# def cal_similarity(request):
#     kw1 = request.GET.get('keyword_1')
#     kw2 = request.GET.get('keyword_2')
#     similarity = semantic_related.get_similarity(kw1, kw2)
#     content = {}
#     content['status'] = similarity is not None
#     if content['status']:
#         content['distance'] = 1 - similarity
#     return JsonResponse(content, safe=False)