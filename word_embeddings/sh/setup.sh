#! /bin/bash

# Add Dependency-Based Wording tools
if [ ! -d "./data/tools" ] 
then
    mkdir ./data/tools 
fi

if [ ! -d "./data/outputs" ] 
then
    mkdir ./data/outputs 
fi

if [ ! -d "./data/corpus" ] 
then
    mkdir ./data/corpus 
fi

if [ ! -d "./data/raw_data" ] 
then
    mkdir ./data/raw_data 
fi

if [ ! -d "./data/test_corpus" ] 
then
    mkdir ./data/test_corpus 
fi

cd ./word2vecf
make word2vecf count_and_filter
mv word2vecf ../data/tools
mv count_and_filter ../data/tools