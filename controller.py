from datasets import *
from features import *

if __name__ == "__main__":
  thead, th, trows = readFile('data/train_rel_2.tsv')
  lhead, lh, lrows = readFile('data/public_leaderboard_rel_2.tsv')
  for i in essaySets():
    print '''
    Building features for essay set %(set)2i
    ==================================
    ''' % {'set': i}

    # Get train and lb rows for this essay set
    strows = essaySet(i, trows, th)
    slrows = essaySet(i, lrows, lh)

    #TODO: split cv here

    # Get all essays for train+lb
    all_essays =  essayVec(strows, th)
    all_essays += essayVec(slrows, lh) # TODO: Try without this

    # Create corpus with all_essays
    w, my_corpus = getCorpus(all_essays)

    # Get features for train and lb
    tfh, tfeatures, _ = getFeatures(strows, my_corpus, th, w, ['EssaySet', 'Score1'])
    lfh, lfeatures, _ = getFeatures(slrows, my_corpus, lh, w, ['EssaySet'])
