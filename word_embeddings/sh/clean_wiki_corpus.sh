#! /bin/bash

# Requirement: 
# 1. The text file has been cleaned by "clean_corpus.sh"
# 2. The text file has been tokenized to sentence level by "my_sent_tokenize.py"

# This file will remove the "== xxxx ==" headers in the wikipedia page content
sed "s/==.*==//g;s/^[ \t]*//g" $1 | tr -s [:space:] > $2
