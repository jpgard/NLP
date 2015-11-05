import math
import nltk
import time

#added by JG - need for defaultdict
from collections import defaultdict

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    unigram_p = {}
    bigram_p = {}
    trigram_p = {}
    corpus = training_corpus
    #preprocess corpus
    
    unigram_sentences = [sentence.replace('\r\n', STOP_SYMBOL).strip() for sentence in corpus]
    
    #unigrams
    unigrams = [each for i in unigram_sentences for each in i.split(' ')]
    unigram_count = defaultdict(int)
    
    for each in unigrams:
        unigram_count[each] += 1
	
    for each in unigram_count:
        unigram_p[(each,)] = math.log(float(unigram_count[each])/len(unigrams),2)
	
    	#bigrams
    bigram_sentences = 	[START_SYMBOL + ' ' + sentence.replace('\r\n', STOP_SYMBOL).strip() for sentence in corpus]
    sentence_bigrams = [list(nltk.bigrams(sentence.split())) for sentence in bigram_sentences]
    bigrams = [bigram for sentence in sentence_bigrams for bigram in sentence]
    bigram_count = defaultdict(int)
    
    for each in bigrams:
        bigram_count[each] += 1
	
    for each in bigram_count:
        if each[0] != '*':
            bigram_p[each] = math.log(float(bigram_count[each])/unigram_count[each[0]],2)
        if each[0] == '*':
            bigram_p[each] = math.log(float(bigram_count[each])/len(bigram_sentences),2)
    		
    	#trigrams
    trigram_sentences = [START_SYMBOL + ' ' + START_SYMBOL + ' ' + sentence.replace('\r\n', STOP_SYMBOL + ' ' + STOP_SYMBOL).strip() for sentence in corpus]
    sentence_trigrams = [list(nltk.trigrams(sentence.split())) for sentence in trigram_sentences]
    trigrams = [trigram for sentence in sentence_trigrams for trigram in sentence]
    trigram_count = defaultdict(int)

    for each in trigrams:
        trigram_count[each] += 1
	
    for each in trigram_count:    
        if each[0] == '*' and each[1] == '*': #handles start characters, end characters shouldn't be a problem
            trigram_p[each] = math.log(float(trigram_count[each])/len(trigram_sentences),2)
        else:
            trigram_p[each] = math.log(float(trigram_count[each])/bigram_count[(each[0], each[1])],2)
    
    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()    
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):

    scores = []
    
    if n==1:
        unigram_sentences = [sentence.replace('\r\n', STOP_SYMBOL).strip() for sentence in corpus]
        unigrams_list = [i.strip().split(' ') for i in unigram_sentences]
        for sentence in unigrams_list:
            try:
                unigram_probs = [ngram_p[(unigram,)] for unigram in sentence]
                scores.append(sum(unigram_probs))
            except KeyError:
                scores.append(MINUS_INFINITY_SENTENCE_LOG_PROB)		
    if n==2:
        bigram_sentences = 	[START_SYMBOL + ' ' + sentence.replace('\r\n', STOP_SYMBOL).strip() for sentence in corpus]
    	bigrams_list = [list(nltk.bigrams(sentence.split(' '))) for sentence in bigram_sentences]
    	
        for sentence in bigrams_list:
            try:
                bigram_probs = [ngram_p[(bigram)] for bigram in sentence]
                scores.append(sum(bigram_probs))
            except KeyError:
                scores.append(MINUS_INFINITY_SENTENCE_LOG_PROB)
                
    if n==3:
        trigram_sentences = [START_SYMBOL + ' ' + START_SYMBOL + ' ' + sentence.replace('\r\n', STOP_SYMBOL + ' ' + STOP_SYMBOL).strip() for sentence in corpus] 
    	trigrams_list = [list(nltk.trigrams(sentence.split(' '))) for sentence in trigram_sentences]
        for sentence in trigrams_list:
            try:
                trigram_probs = [ngram_p[(trigram)] for trigram in sentence]
                scores.append(sum(trigram_probs))
            except KeyError:
                scores.append(MINUS_INFINITY_SENTENCE_LOG_PROB)
    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    l = float(1)/3
    #preprocessing
    
    corpus_pr = [START_SYMBOL + ' ' + START_SYMBOL + ' ' + i for i in corpus]
    corpus_pr = [sentence.replace('\r\n', STOP_SYMBOL).strip() for sentence in corpus_pr]
  
    #tokenize sentences
    sent_tokens = [sentence.split(' ') for sentence in corpus_pr]
    
    
    for sentence in sent_tokens:
        sentence_p = 0
        for trigram in list(nltk.trigrams(sentence)):
            try:
                sentence_p += math.log(l*(2**trigrams[trigram] + 2**bigrams[(trigram[1], trigram[2])] + 2**unigrams[(trigram[2],)]),2)
            except KeyError:
                scores.append(MINUS_INFINITY_SENTENCE_LOG_PROB)
                continue
        scores.append(sentence_p)
    
    
    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
