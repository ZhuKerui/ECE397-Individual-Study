import csv
import json
import io
import requests
from wikidata.client import Client
from wikidata.entity import Entity

def find_id(keyword_file, output_file, error_log):
    with io.open(keyword_file, 'r', encoding='utf-8') as load_file:
        with io.open(output_file, 'w', encoding='utf-8') as dump_file:
            with io.open(error_log, 'w', encoding='utf-8') as log_file:
                f_csv = csv.writer(dump_file)
                url = 'https://www.wikidata.org/w/api.php'
                params = {'action':'wbsearchentities','format':'json','language':'en'}
                cnt = 0
                for word in load_file:
                    word = word.strip()
                    params['search'] = word
                    r = requests.get(url, params=params)
                    if r.json()['search']:
                        id = r.json()['search'][0]['id']
                        label = r.json()['search'][0]['label']
                        f_csv.writerow([word, id, label])
                    else:
                        log_file.write(word + '\n')
                    cnt += 1
                    if cnt % 100 == 0:
                        print(cnt)

# instance of(P31), subclass of(P297), part of(P361), facet of(P1269), manifestation of(P1557)
def find_related(keyword_csv, related_set):
    with io.open(keyword_csv, 'r', encoding='utf-8') as load_file:
        with io.open(related_set, 'w', encoding='utf-8') as dump_file:
            load_csv = csv.reader(load_file)
            client = Client()
            instance_of = Entity('P31', client)
            subclass_of = Entity('P279', client)
            part_of = Entity('P361', client)
            facet_of = Entity('P1269', client)
            manifestation_of = Entity('P1557', client)
            myDict = {}
            cnt = 0
            for row in load_csv:
                id = row[1]
                entity = client.get(id, load=True)
                myDict[id] = {}
                myDict[id]['P31'] = [item.id for item in entity.getlist(instance_of)]
                myDict[id]['P279'] = [item.id for item in entity.getlist(subclass_of)]
                myDict[id]['P361'] = [item.id for item in entity.getlist(part_of)]
                myDict[id]['P1269'] = [item.id for item in entity.getlist(facet_of)]
                myDict[id]['P1557'] = [item.id for item in entity.getlist(manifestation_of)]
                cnt += 1
                if cnt % 100 == 0:
                    print(cnt)
            json.dump(myDict, dump_file)