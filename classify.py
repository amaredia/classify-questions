# gensim modules
from gensim.models import Word2Vec
import numpy as np
import os

stopwords = ['a', 'to', 'and', 'of', 'um', 'mkay', 'okay', 'uh', 'um', 'er', 
'ah', 'eh', 'oh', 'so','haha', 'mm', 'how', 'whom', 'who', 'what', 'do', 'did',
'have', 'you', 'ever', 'is', 'really', 'for', 'your', 'the', 'were', 'was',
'ha', '#', 'where', 'in', 'into', 'whats', 'from', 'on', 'it', 'or']

q_words = ['what', 'how', 'who', 'when', 'where', 'whom', 'did', 'have', 'do']

weighted = ['born', 'years', 'live', 'home', 'mothers', 'fathers', 'parents',
'job', 'divorced', 'divorce', 'broken', 'bone', 'allergies',
'food', 'foods', 'overnight', 'hospital', 'tweeted', 'bought', 'ebay', 'ereader', 
'physical', 'fight', 'trouble', 'police', 'romantic', 'relationship', 'love', 
'spent','shoes', 'movie', 'hated', 'ice', 'skating', 'tennis', 'racket', 
'roommates','major', 'cat', 'die', 'cheat', 'cheated', 'test', 'high', 'school']
        
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
            print stripped
            
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
    mismatch = 0
    uncaught = 0
    
    #calculates total sum of all the differences between actual vector average
    #of question and calculated vector average of turn
    cor_diff = 0
    incor_diff = 0
    cor_max_thresh = float("-inf")
    incor_min_thresh = float("inf")
    
    #goes through lines of every file
    q_score = convertToDict('q_vect.txt')
    for line in lines:
        cleaned = list()
        sentence = line.split(",")
        question = sentence[4]
        
        #cleans each question
        question = question.lower()
        question = question.replace("iceskating", "ice skating")
        question = question.replace("e-reader", "ereader")
        question = question.replace("e reader", "ereader")
        question = question.replace(" got ", " gotten ")
        question = question.replace("allergy", "allergies")
        question = question.replace(" tweet ", " tweeted ")
        question = question.replace("food", "foods")
        question = question.replace("'s", "s")
        question = question.split(" ")
        question = [word for word in question if "-" not in word]
        question = [word for word in question if "'" not in word]
        
        #filters out only just after a question word
        i = 0
        is_q = False
        prefiltered = list(question)
        while i< len(prefiltered) and is_q == False:
            if prefiltered[i] in q_words:
                is_q = True
            else:
                if prefiltered[i] not in weighted:
                    question.remove(prefiltered[i])
            i = i + 1
        
        #filters stop words and repeats
        question  = [word for word in question if word not in stopwords]
        for word in question:
            if word not in cleaned:
                cleaned.append(word)
        
        q_match = sentence[5]
        if len(cleaned) != 0:
            q_avg = assignVectorAvg(cleaned)
            closestMatchQ, closestMatchV = min(q_score.items(), key=lambda (_, v): abs(v - q_avg))
            
            #determines whether a question is correctly matched or not
            if "f" not in q_match and q_match != "0":
                q_match = q_match.replace("/", "")
                diff = abs(closestMatchV-q_avg)
                try:
                    if closestMatchQ == int(q_match):
                        correct = correct + 1
                        cor_diff = cor_diff + diff
                        if diff > cor_max_thresh:
                            cor_max_thresh = diff
                        #print str(sentence[0]) + " " + str(q_avg) + " " + str(closestMatchQ) + " " + str(closestMatchV) + " correct " + str(abs(closestMatchV - q_avg))

                    else:
                        incorrect = incorrect + 1
                        incor_diff = incor_diff + diff
                        if diff < incor_min_thresh:
                            incor_min_thresh = diff
                        #print str(sentence[0]) + " " + str(q_avg) + " " + str(closestMatchQ) + " " + str(closestMatchV) + " incorrect"
                        #print cleaned
                except ValueError:
                    continue
            else:
                mismatch = mismatch + 1
        elif "f" not in q_match and q_match != "0":
            #print str(sentence[0])
            uncaught = uncaught + 1
    #print str(cor_max_thresh) + " " + str(incor_min_thresh)
    print fileName + " " + str(correct) + " " + str(incorrect) + " " + str(mismatch) + " " + str(uncaught)
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
    #parseRawQFile("questions.csv")
    #classifyTurns("labeled/p362p363-part2_ch1.csv")
    for filename in os.listdir("labeled/"):
        if filename.endswith(".csv"):
            name = "labeled/" + filename
            classifyTurns(name)

if __name__ == "__main__":
    main()
    
