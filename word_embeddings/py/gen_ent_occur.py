# python gen_ent_occur.py entity_file sentence_file occur_file
import sys

sys.path.append('..')
from tools.BasicUtils import MultiProcessing
from tools.TextProcessing import occurance_dump, occurance_post_operation, Entity_Occurance

# Generate occurance file
p = MultiProcessing()
occur_dict = p.run(lambda: Entity_Occurance(sys.argv[1]), open(sys.argv[2]).readlines(), 8, occurance_post_operation)
occurance_dump(sys.argv[3], occur_dict)