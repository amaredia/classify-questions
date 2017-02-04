# classify-questions

##identify_vect.py

This file uses Word2Vec to find vector averages for every question and turn. Using Word2Vec, we are able to find a word vector for every word in each question and turn. These word vectors are averaged across the entire question/ turn to find the vector average. Certain stopwords have been filtered out, some spelling errors have been adjusted, and certain words have associated weights to indicate their performance. 

Once these vector averages have been found for each question, we find the vector averages for each turn. We then find the cosine similarity between the turn vector average and the question vector average. The turn with the highest cosine similarity to a question out of all the turns will be identified as that particular question.

###identify_vect_API.py

This approach utilizes this [API](http://swoogle.umbc.edu/SimService/api.html) and [STS Similarity](http://swoogle.umbc.edu/StsService/index.html). This returns the cosine similarity of the semantic vectors formed in each question and turn.

##How to run and test

To test this program, simply call classifyTurns(filename) passing in the name of the .csv file that contains the turns in main(). Right now, the program outputs at the end of classifyTurns how many questions were identified correctly and how many were incorrectly identified.
