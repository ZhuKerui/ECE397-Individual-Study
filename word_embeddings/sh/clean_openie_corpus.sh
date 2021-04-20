#! /bin/bash

# Requirement: 
# 1. The text file has been cleaned by "clean_corpus.sh"
# 2. The text file has been tokenized to sentence level by "my_sent_tokenize.py"
# 3. If the text file is from wikipedia page content, it should have been cleaned by "clean_wiki_corpus.sh"

# This file will remove the lines containing word "null", which will cause trobule when operating ollie
sed -i "/null/d" $1