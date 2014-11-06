# code from
# http://sujitpal.blogspot.com/2013/02/checking-for-word-equivalence-using.html
from __future__ import division
from nltk.corpus import wordnet as wn
import itertools as it
import sys

# similarity between two vectors as comparing
# all non-stop words, normalized to be between 0 and 1
# input is two vector tokens w/o stopwords
def vec_semantic_sim(v1,v2):
    sim_acc = 0.0
    compares = 0 
    for (w1, w2) in it.product(v1, v2):
        sim_acc += similarity(w1,w2)
        compares += 1
    return sim_acc/ float(compares)
         
    
def similarity(w1, w2, sim=wn.path_similarity):
  synsets1 = wn.synsets(w1)
  synsets2 = wn.synsets(w2)
  sim_scores = []
  for synset1 in synsets1:
    for synset2 in synsets2:
      sim_scores.append(sim(synset1, synset2))
  if len(sim_scores) == 0:
    return 0
  else:
    return max(sim_scores)

def main():
    (word1, word2) = ("large", "big")
    print(similarity(word1, word2))

if __name__ == "__main__":
  main()
