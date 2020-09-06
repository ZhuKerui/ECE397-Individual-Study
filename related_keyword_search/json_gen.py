import os
import io
import requests
import json
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

def json_gen(word_list_path, json_store_path):
    # if os.path.exists(word_list_path) and not os.path.exists(json_store_path):
    if os.path.exists(word_list_path):
        results = {}
        params = {"client":"firefox", "hl":"en"}
        url = 'http://suggestqueries.google.com/complete/search'
        with io.open(word_list_path, 'r', encoding='utf-8') as load_file:
            for line in load_file:
                q_word = line.strip().lower()
                if q_word:
                    params["q"] = q_word
                    temp_r = requests.get(url, params=params)
                    result = temp_r.json()
                    # Make sure that the keyword itself is in the list
                    keyword_list = q_word.split(' ')
                    for suggest in result[1]:
                        # Skip the results that change the query word
                        if q_word not in suggest:
                            continue
                        related_words = suggest.split(' ')
                        for word in related_words:
                            # if word not in keyword_list and word not in stopwords.words():
                            if word not in keyword_list:
                                keyword_list.append(word)
                    results[q_word] = keyword_list
        with io.open(json_store_path, 'w', encoding='utf-8') as dump_file:
            json.dump(results, dump_file)
