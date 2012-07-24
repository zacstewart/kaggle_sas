# The Hewlett Foundation: Short Answer Scoring

Hey look I'm doing a thing! It's in R and Python. I'm using NLTK in Python
to generate features and R to classify. I'm currently using a random forest
to do that.

> Develop a scoring algorithm for student-written short-answer responses.

## How to do it
1. Generate features: run _features.py_ with arguments trainfile, testfile and
   columns to copy from each to their respective feature files.  
   `python features.py data/train.tsv data/test.tsv EssaySet Score1`
2. Fire up R and do `source('project.r')`
