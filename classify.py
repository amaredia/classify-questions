# gensim modules
from gensim import utils
from gensim.models import Word2Vec

print "Loading model"
model = Word2Vec.load("GoogleNewsVectors.bin")
print "Model has been loaded"