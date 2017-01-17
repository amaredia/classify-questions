# gensim modules
from gensim.models import Word2Vec
import numpy as np
import os
import re

stopwords = ['a', 'to', 'and', 'of', 'um', 'mkay', 'okay', 'uh', 'um', 'er', 
'ah', 'eh', 'oh', 'so','haha', 'mm', 'how', 'whom', 'who', 'what', 'do', 'did',
'have', 'you', 'your', 'ever', 'is']

q_words = ['what', 'how', 'who', 'when', 'where', 'whom', 'did', 'have', 'do']

weighted = ['born', 'years', 'live', 'home', 'mothers', 'fathers', 'parents',
'job', 'divorced', 'broken', 'bone', 'allergies', 'food', 'foods',
'overnight', 'hospital', 'tweeted', 'bought', 'ebay', 'ereader', 'physical',
'fight', 'trouble', 'police', 'romantic', 'relationship', 'love', 'spent',
'shoes', 'movie', 'hated', 'ice', 'skating', 'tennis', 'racket', 'roommates',
'major', 'cat', 'die', 'cheat', 'test', 'high', 'school']
        
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
            
            #cleaning up punctuation
            line = line.replace("e-reader", "ereader")
            line = line.replace("-", " ")
            line = line.replace("'", "")
            stripped = line.split(" ")
            
            #filters out only just after a question word
            i = 0
            is_q = False
            prefiltered = list(stripped)
            while i< len(prefiltered) and is_q == False:
                if prefiltered[i] in q_words:
                    is_q = True
                else:
                    stripped.remove(prefiltered[i])
                i = i + 1
            
            #cleans up stopwords
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
    
    #counts how many correct and incorrect matches
    correct = 0
    incorrect = 0
    
    #goes through lines of every file
    q_score = convertToDict('q_vect.txt')
    for line in lines:
        cleaned = list()
        sentence = line.split(",")
        question = sentence[4]
        
        #cleans each question
        question = question.lower()
        question = question.replace("-", "")
        question = question.replace("iceskating", "ice skating")
        question = question.replace("'", "")
        question = question.split(" ")
        
        #filters out only just after a question word
        i = 0
        is_q = False
        prefiltered = list(question)
        while i< len(prefiltered) and is_q == False:
            if prefiltered[i] in q_words:
                is_q = True
            else:
                question.remove(prefiltered[i])
            i = i + 1
        
        #filters stop words and repeats
        question  = [word for word in question if word not in stopwords]
        question.sort()
        cleaned = [word for word in question if word not in cleaned]
        
        if len(question) != 0:
            q_avg = assignVectorAvg(cleaned)
            closestMatchQ, closestMatchV = min(q_score.items(), key=lambda (_, v): abs(v - q_avg))
            
            #determines whether a question is correctly matched or not
            q_match = sentence[5]
            if "f" not in q_match and q_match != "0":
                q_match = q_match.replace("/", "")
                try:
                    if closestMatchQ == int(q_match):
                        correct = correct + 1
                    else:
                        incorrect = incorrect + 1
                except ValueError:
                    continue
                    
            #print str(sentence[0]) + " " + str(q_avg) + " " + str(closestMatchQ) + " " + str(closestMatchV)
    
    print fileName + " " + str(correct) + " " + str(incorrect)
    f.close()
    
    
#Assigns a vector average to a given phrase based on the model and returns average
def assignVectorAvg(phrase):
    sum = 0
    wordCount = 0
    for word in phrase:
        try:
            if word in weighted:
                sum = sum + (model[word]*3)
            else:
                sum = sum + model[word]
            wordCount = wordCount + 1
        except KeyError:
            continue
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
    parseRawQFile("questions.csv")
    for filename in os.listdir("labeled/"):
        name = "labeled/" + filename
        classifyTurns(name)

if __name__ == "__main__":
    main()
    
