# gensim modules
from gensim import utils
from gensim.models import Word2Vec
import os
import numpy as np

#Loading model
print "Loading model"
model = Word2Vec.load_word2vec_format("GoogleNewsVectors.bin", binary=True)
print "Model has been loaded"


#Parses through each question and write question and vector average to file
def parseRawQFile(fileName):
    
    #open file for reading and writing
    f = open(fileName)
    lines = filter(None, (line.rstrip() for line in f))
    q = open('q_vect.txt', 'w')
    qNum = 1 #set question number
    
    #parse through each line
    for line in lines:
        if line != "Questions":
            line = line.lower()
            
            #cleaning up punctuation and stopwords
            line = line.replace("-", " ")
            line = line.replace("'", "")
            line = line.replace(" a ", " ")
            line = line.replace(" to ", " ")
            line = line.replace(" and ", " ")
            line = line.replace(" of ", " ")
            stripped = line.split(" ")
            
            #determine average of the phrase
            avg = assignVectorAvg(stripped)
            q.write(str(qNum) + " " + str(avg) + "\n")
            qNum = qNum + 1
    
    #close files
    f.close()
    q.close()
    
#Parses through file for each turn and writes to file
#def classifyTurns(fileName):
    
#Assigns a vector average to a given phrase based on the model and returns average
def assignVectorAvg(phrase):
    sum = 0
    wordCount = 0
    for word in phrase:
        sum = sum + model[word]
        wordCount = wordCount + 1
    avg = sum/wordCount
    return np.average(avg)
    
#Finds closest question and returns number or null
#def matchQuestion():
    
#Converts raw question vector file into dict and returns dict
#def convertToDict():

def main():
    parseRawQFile("questions.csv")

if __name__ == "__main__":
    main()
    
