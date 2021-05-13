# python gen_occur.py wordtree_file keyword_file sentence_file occur_file
import sys

sys.path.append('..')
from tools.BasicUtils import MultiProcessing
from tools.DocProcessing.Occurrence import Occurrence, occurrence_dump, occurrence_post_operation

# Generate occurrence file
p = MultiProcessing()
occur_dict = p.run(lambda: Occurrence(sys.argv[1], sys.argv[2]), open(sys.argv[3]).readlines(), 8, occurrence_post_operation)
occurrence_dump(sys.argv[4], occur_dict)