# gensim modules
from gensim.models import Word2Vec
import numpy as np
import math
import os

#TODO: 
#    -Questions separated by small pauses (what is that threshold)
#    -Questions that have more than one label (what do your parents do)
#    -Questions that depend on context
#    -Incorrect matches are matches that are followup questions to an original Q

stopwords = ['a', 'to', 'and', 'of', 'um', 'mkay', 'okay', 'uh', 'um', 'er', 
'ah', 'eh', 'oh', 'so','haha', 'mm', 'you', 'ever', 'is', 'really', 'for', 'your', 
'the', 'were', 'was', 'ha', '#', 'in', 'into', 'from', 'on', 'it', 'or']

weighted = ['born', 'years', 'live', 'home', 'mothers', 'fathers', 'parents',
'job', 'divorced', 'divorce', 'broken', 'bone', 'allergies',
'food', 'foods', 'overnight', 'hospital', 'tweeted', 'bought', 'ebay', 'ereader', 
'physical', 'fight', 'trouble', 'police', 'romantic', 'relationship', 'love', 
'spent','shoes', 'movie', 'hated', 'ice', 'skating', 'tennis', 'racket', 
'roommates','major', 'cat', 'die', 'cheat', 'cheated', 'test', 'high', 'school',
'tweet']

#Loading model
print "Loading model"
model = Word2Vec.load_word2vec_format("GoogleNewsVectors.bin", binary=True)
print "Model has been loaded"

#Parses through each question and write question and vector average to file
def createQVects(fileName):
    q_vect = {}
    
    #open file for reading and writing
    f = open(fileName, 'r')
    lines = filter(None, (line.rstrip() for line in f))
  
    q_num = 1 #set question number
    
    #parse through each line
    for line in lines:
        if line != "Questions":
            line = line.lower()
            
            #cleaning up punctuation
            line = line.replace("e-reader", "ereader")
            line = line.replace("-", " ")
            line = line.replace("'", "")
            stripped = line.split(" ")
            
            #cleans up stopwords
            stripped  = [word for word in stripped if word not in stopwords]
            
            #determine average of the phrase
            avg = assignVectorAvg(stripped)
            q_vect[q_num] = avg
            q_num = q_num + 1
    
    #close files
    f.close()
    
    #return vectors
    return q_vect
    
    
#Parses through file for each turn and writes to file
def classifyTurns(fileName):
    
    final_q = {}
    q_vect = createQVects("questions.csv")
        
    #clean text by removing spaces
    f = open(fileName, 'r')
    lines = filter(None, (line.rstrip() for line in f))
    lines.pop(0)
    
    #counts how many correct and incorrect matches
    correct = 0
    incorrect = 0
    uncaught = 0
    
    #goes through lines of every file
    for line in lines:
        cleaned = list()
        sentence = line.split(",")
        turn = sentence[4]
        
        #cleans each question
        turn = turn.lower()
        turn = turn.replace("iceskating", "ice skating")
        turn = turn.replace("e-reader", "ereader")
        turn = turn.replace("e reader", "ereader")
        turn = turn.replace(" got ", " gotten ")
        turn = turn.replace("allergy", "allergies")
        turn = turn.replace(" tweet ", " tweeted ")
        turn = turn.replace("food", "foods")
        turn = turn.replace("'s", "s")
        turn = turn.split(" ")
        turn = [word for word in turn if "-" not in word]
        turn = [word for word in turn if "'" not in word]
        
        #filters stop words and repeats
        turn  = [word for word in turn if word not in stopwords]
        for word in turn:
            if word not in cleaned:
                cleaned.append(word)
        
        turn_match = sentence[5]
        turn_match = turn_match.split(" ")
        turn_match = turn_match[0]
        cos_similar = {}
        if len(cleaned) != 0:
            turn_avg = assignVectorAvg(cleaned)
            if turn_avg is None:
                continue
            
            #Calculates cosine similarity for each question
            for q_num in q_vect.keys():
                dot_product = np.dot(turn_avg, q_vect[q_num])
                mag_turn = math.sqrt(np.dot(turn_avg, turn_avg))
                mag_q = math.sqrt(np.dot(q_vect[q_num], q_vect[q_num]))
                cos_similar[q_num] = dot_product/(mag_turn * mag_q)
            
            #Compares top two question matches to find matches
            closest_match_q = max(cos_similar, key=cos_similar.get)
            if closest_match_q not in final_q or cos_similar[closest_match_q] > final_q[closest_match_q][0]:
                final_q[closest_match_q] = [cos_similar[closest_match_q], sentence]
            else:
                cos_similar[closest_match_q] = 0;
                closest_match_q = max(cos_similar, key=cos_similar.get)
                if closest_match_q not in final_q or cos_similar[closest_match_q] > final_q[closest_match_q][0]:
                    final_q[closest_match_q] = [cos_similar[closest_match_q], sentence]

        elif "f" not in sentence[5] and turn_match != "0":
            uncaught = uncaught + 1
        
        
    #determines whether a question is correctly matched or not
    for question in final_q:
        sentence = final_q[question][1]
        turn_match = sentence[5]
        turn_match = turn_match.split(" ")
        turn_match = turn_match[0]
        if "f" not in turn_match and turn_match != "0":
            turn_match = turn_match.replace("/", "")
            try:
                if question == int(turn_match):
                    correct = correct + 1
                else:
                    incorrect = incorrect + 1
            except ValueError:
                continue
        else:
            incorrect = incorrect + 1
    print fileName + " " + str(correct) + " " + str(incorrect) + " " + str(uncaught)
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
        return None
    
    avg = sum/wordCount
    return avg
            

def main():
    #classifyTurns("labeled/p212p213-part1_ch2.csv")
    for filename in os.listdir("labeled/"):
        if filename.endswith(".csv"):
            name = "labeled/" + filename
            classifyTurns(name)
    
if __name__ == "__main__":
    main()
    
