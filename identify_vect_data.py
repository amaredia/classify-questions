# gensim modules
from gensim.models import Word2Vec
import numpy as np
import math
import csv
import sys

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

mom_words = ['mom', 'mom\'s', 'mother', 'mother\'s', 'mothers'] 
dad_words = ['dad', 'dad\'s', 'father', 'father\'s', 'fathers']

all_turns = {}

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
def classifyTurns(fileName, lines):
    
    final_q = {}
    q_vect = createQVects("questions.csv")
    mom_turns = []
    dad_turns = []
              
    #goes through lines of every file
    for line in lines:
        cleaned = list()
        sentence = line
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
        
        #stores each turns cosine similarities to a question
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
            
            if set.intersection(set(cleaned), set(mom_words)):
                if "mother do" in sentence[4] or "mom do" in sentence[4] or "about" in sentence[4]:
                    mom_turns.append((cos_similar[3], sentence))
            
            if set.intersection(set(cleaned), set(dad_words)):
                if "father do" in sentence[4] or "dad do" in sentence[4] or "about" in sentence[4]:
                    dad_turns.append((cos_similar[4], sentence))
        
    #adjust for rule when mom and dad job question keeps on getting missed
    if 3 not in final_q:
        if len(mom_turns) > 0:
            mom_turns.sort(reverse=True)
            final_q[3] = mom_turns[0]
    
    if 4 not in final_q:
        if len(dad_turns) > 0:
            dad_turns.sort(reverse=True)
            final_q[4] = dad_turns[0]

    global all_turns            
    for num in final_q:
        all_turns[np.float64((final_q[num])[0]).item()] = (num, (final_q[num])[1])
    
    #Gives all matched lines with corresponding question number
    matched_lines = {}
    for num in final_q.keys():
        q_cos = final_q[num]
        val = ','.join([str(q) for q in q_cos[1]])
        matched_lines[val] = num

    #Writes output 
    for line in lines:
        #line.insert(0, fileName)
        q_detected = 0
        key = ','.join([str(word) for word in line])
        if key in matched_lines.keys():
            q_detected = matched_lines[key]
        line.append(q_detected)
        
        with open('ER_q_annotations.csv', 'a') as csvfile:
            q_writer = csv.writer(csvfile, delimiter='\t',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            q_writer.writerow(line)
        
    print fileName + ": identified " + str(len(final_q)) + " questions"
    
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
    #clean text by removing spaces
    lines = [line for line in csv.reader(open('interviewer_turns.csv', 'r'),delimiter='\t')]
    header = lines.pop(0)
    header.append('question')
    
    with open('ER_q_annotations.csv', 'a') as csvfile:
        q_writer = csv.writer(csvfile, delimiter='\t',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        q_writer.writerow(header)

    #divides turns into respective files to be classified
    fileName = lines[0][1]
    chunk = []
    for line in lines:
        if line[1] == fileName:
            chunk.append(line)
        else:
            classifyTurns(fileName, chunk)
            chunk = []
            fileName = line[1]
    classifyTurns(fileName, chunk)
        
    global all_turns
    keys = sorted(all_turns)
    keys = keys[0:2000]
    for key in keys:
        print str(key) + " {0}".format(all_turns[key])

    
if __name__ == "__main__":
    main()
    
