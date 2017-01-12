# gensim modules
from gensim.models import Word2Vec
import numpy as np

stopwords = ['a', 'to', 'and', 'of', 'um', 'mkay', 'okay', 'uh', 'um', 'er', 
'ah', 'eh', 'oh', 'so','haha', 'mm', 'how', 'whom', 'who', 'what', 'do', 'did']
    
#Loading model
print "Loading model"
model = Word2Vec.load_word2vec_format("GoogleNewsVectors.bin", binary=True)
print "Model has been loaded"


#Parses through each question and write question and vector average to file
def parseRawQFile(fileName):
    
    #open file for reading and writing
    f = open(fileName, 'r')
    lines = filter(None, (line.rstrip() for line in f))
    q = open('q_vect.txt', 'w')
    qNum = 1 #set question number
    
    #parse through each line
    for line in lines:
        if line != "Questions":
            line = line.lower()
            
            #cleaning up punctuation and stopwords
            line = line.replace("e-reader", "ereader")
            line = line.replace("-", " ")
            line = line.replace("'", "")
            stripped = line.split(" ")
            stripped  = [word for word in stripped if word not in stopwords]
            
            #determine average of the phrase
            avg = assignVectorAvg(stripped)
            q.write(str(qNum) + " " + str(avg) + "\n")
            qNum = qNum + 1
    
    #close files
    f.close()
    q.close()
    
    
#Parses through file for each turn and writes to file
def classifyTurns(fileName):
    #clean text using stop words
    f = open(fileName, 'r')
    lines = filter(None, (line.rstrip() for line in f))
    lines.pop(0)
    
    q_score = convertToDict('q_vect.txt')
    #initialize array of questions with yes or no
    for line in lines:
        cleaned = list()
        sentence = line.split(",")
        question = sentence[4]
        
        question = question.lower()
        question = question.replace("-", "")
        question = question.replace("iceskating", "ice skating")
        question = question.replace("'", "")
        question = question.split(" ")
        question  = [word for word in question if word not in stopwords]
        
        question.sort()
        
        cleaned = [word for word in question if word not in cleaned]
        
        if len(question) != 0:
            q_avg = assignVectorAvg(cleaned)
            closestMatchQ, closestMatchV = min(q_score.items(), key=lambda (_, v): abs(v - q_avg))
            print str(sentence[0]) + " " + str(q_avg) + " " + str(closestMatchQ) + " " + str(closestMatchV)
    
    f.close()
    
    
#Assigns a vector average to a given phrase based on the model and returns average
def assignVectorAvg(phrase):
    sum = 0
    wordCount = 0
    for word in phrase:
        try:
            sum = sum + model[word]
            wordCount = wordCount + 1
        except KeyError:
            print word
    if wordCount == 0:
        return 0
    
    avg = sum/wordCount
    return np.average(avg)


#Converts raw question vector file into dict and returns dict
def convertToDict(fileName):
    question_avg = {}
    f = open(fileName,'r')
    lines = filter(None, (line.rstrip() for line in f))
    for line in lines:
        q_score = line.split(" ")
        q_score[0] = int(q_score[0])
        q_score[1] = float(q_score[1])
        question_avg[q_score[0]] = q_score[1]
    return question_avg
        

def main():
    #parseRawQFile("questions.csv")
    classifyTurns("labeled/p364p365-part2_ch1.csv")

if __name__ == "__main__":
    main()
    
