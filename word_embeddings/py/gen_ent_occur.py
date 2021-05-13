# python gen_ent_occur.py entity_file sentence_file occur_file
import sys

sys.path.append('..')
from tools.BasicUtils import MultiProcessing
from tools.DocProcessing.EntityOccurrence import EntityOccurrence
from tools.DocProcessing.Occurrence import occurrence_dump, occurrence_post_operation

# Generate occurrence file
p = MultiProcessing()
occur_dict = p.run(lambda: EntityOccurrence(sys.argv[1]), open(sys.argv[2]).readlines(), 8, occurrence_post_operation)
occurrence_dump(sys.argv[3], occur_dict)