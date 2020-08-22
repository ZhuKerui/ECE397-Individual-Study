import MyEngine

engine = MyEngine.MyEngine('../../../test_index', '../../../dataset/snapshot.json')
print(engine.search('2008-12-13','',['abstract'], True))