import A
from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict, Counter
import nltk
import string
from sklearn import svm
from sklearn import neighbors
from nltk.corpus import stopwords
import math
from nltk.corpus import wordnet as wn
import sys
from nltk.data import load
import nltk.tag
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2



# You might change the window size
window_size = 3

# B.1.a,b,c,d
def extract_features(data):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
            [(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :return: features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }
    '''
    
    features = {}
    labels = {}
    instance_windowtokens = {}
    window_counts = {}
    s = []
    nc = {}
    nsc = {}
    
    # implement your code here
    
    #ENGLISH LOOP
        
    if sys.argv[-1] == 'English':
    
        # print 'ENGLISH'
        _POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
        tagger = load(_POS_TAGGER)
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        
        #first pass for relevancy score counts
#         
#         for inst in data:
#             
#             instance_id = inst[0]
#             lcontext = inst[1]
#             head = inst[2]
#             rcontext = inst[3]
#             sense_id = inst[4]
#             
#             left_context_all = nltk.word_tokenize(lcontext)
#             left_context_tokens = [token for token in left_context_all if not token in string.punctuation] #and not token in stopwords.words('english')
#             left_context_tokens = left_context_tokens[-window_size:]
#             
#             right_context_all = nltk.word_tokenize(rcontext)
#             right_context_tokens = [token for token in right_context_all if not token in string.punctuation] # and not token in stopwords.words('english')
#             right_context_tokens = right_context_tokens[0:window_size]
#         
#             context = left_context_tokens + right_context_tokens 
#             
#             
#             # counts for relevance score
#             
#             window_size=5
#        
#             left =left_context_all[(-window_size):]
#             right=right_context_all[0:window_size]
#        
#             unique_context= set(left+right)
#        
#             for item in unique_context:
#             if item not in nc.keys():
#                nc[item]=1
#            
#             else:
#                 nc[item]=nc[item]+1
#            
#            if sense_id not in nsc.keys():
#                nsc[sense_id]={}
#            
#            if item not in nsc[sense_id].keys():
#                nsc[sense_id][item]=1
#            
#            else:
#                nsc[sense_id][item] +=1
#             
#  
        #second pass, now using counts to compute relevance score
        
        for inst in data:
        
            instance_id = inst[0]
            lcontext = inst[1]
            head = inst[2]
            rcontext = inst[3]
            sense_id = inst[4]
        
                
            features[instance_id] = {}
            
            
        
            if instance_id and sense_id: 
                labels[instance_id] = sense_id
        
            left_context_all = nltk.word_tokenize(lcontext)
            left_context_tokens = [token for token in left_context_all if not token in string.punctuation] #and not token in stopwords.words('english')
            left_context_tokens = left_context_tokens[-window_size:]
            
            right_context_all = nltk.word_tokenize(rcontext)
            right_context_tokens = [token for token in right_context_all if not token in string.punctuation] # and not token in stopwords.words('english')
            right_context_tokens = right_context_tokens[0:window_size]
        
            context = left_context_tokens + right_context_tokens 
            
            
            
            
            
    #         instance_windowtokens[instance_id] = context
        
            #surrounding words, no stopwords or punctuation, with stemming
        
            for i in range(-3,0):
                try: features[instance_id]['w_' + str(-i)] = SnowballStemmer("english").stem(left_context_tokens[i])
                except IndexError: features[instance_id]['w_' + str(-i)] = ''
        
            for i in range(0,3):
                try: features[instance_id]['w' + str(i)] = SnowballStemmer("english").stem(right_context_tokens[i])
                except IndexError: features[instance_id]['w' + str(i)] = ''
        
            #collocations
        
            left_tokens_all = [token for token in left_context_all if not token in string.punctuation]
            right_tokens_all = [token for token in right_context_all if not token in string.punctuation]
        
            try: features[instance_id]['c-3-2-1'] = left_tokens_all[-3] + '_' + left_tokens_all[-2] + '_' + left_tokens_all[-1]
            except IndexError: features[instance_id]['c-3-2-1'] = ''
        
            try: features[instance_id]['c-2-2'] = left_tokens_all[-2]
            except IndexError: features[instance_id]['c-2-2'] = ''
        
            try: features[instance_id]['c-2-1'] = left_tokens_all[-2] + '_' + left_tokens_all[-1]
            except IndexError: features[instance_id]['c-2-1'] = ''
        
            try: features[instance_id]['c-1-1'] = left_tokens_all[-1]
            except IndexError: features[instance_id]['c-1-1'] = ''
        
            try: features[instance_id]['c-2_1'] = left_tokens_all[-2] + '_' + left_tokens_all[-1] + right_tokens_all[0]
            except IndexError: features[instance_id]['c-2_1'] = ''
        
            try: features[instance_id]['c11'] = right_tokens_all[0]
            except IndexError: features[instance_id]['c11'] = ''
        
            try: features[instance_id]['c12'] = right_tokens_all[0] + '_' + right_tokens_all[1]
            except IndexError: features[instance_id]['c12'] = ''
        
            try: features[instance_id]['c22'] = right_tokens_all[1]
            except IndexError: features[instance_id]['c22'] = ''
        
            try: features[instance_id]['c-1_1'] = left_tokens_all[-1] + '_' + right_tokens_all[0]
            except IndexError: features[instance_id]['c-1_1'] = ''
        
            try: features[instance_id]['c-1_2'] = left_tokens_all[-1] + '_' + right_tokens_all[0] + '_' + right_tokens_all[1]
            except IndexError: features[instance_id]['c-1_2'] = ''
        
            try: features[instance_id]['c123'] = right_tokens_all[0] + '_' + right_tokens_all[1] + '_' + right_tokens_all[2]
            except IndexError: features[instance_id]['c123'] = ''

            #synset - top 3 synonyns of head
            syns = wn.synsets(instance_id.split('.')[0], instance_id.split('.')[1])
            i = 0
            for each in syns[0:2]:
                if each:
                    features[instance_id]['head_syn_' + str(i)] = each.name()
                    i += 1
    #         
    #         #synset - top 3 synonyns of w-2
    #         try:
    #           syns = wn.synsets(left_context_tokens[-2])
    #           i = 0
    #           for each in syns[0:2]:
    #               if each:
    #                   features[instance_id]['w-2_syn_' + str(i)] = each.name()
    #                   i += 1
    #         except IndexError:
    #             i = 0
    #             for each in range(0,3):
    #               features[instance_id]['w-2_syn_' + str(i)] = ''
    #               i += 1
        
            #synset - top 3 synonyns of w-1
            try:
                syns = wn.synsets(left_context_tokens[-1])
                i = 0
                for each in syns[0:2]:
                    if each:
                        features[instance_id]['w-1_syn_' + str(i)] = each.name()
                        i += 1
            except IndexError:
                i = 0
                for each in range(0,3):
                    features[instance_id]['w-1_syn_' + str(i)] = ''
                    i += 1
        
        
        
            #synset - top 3 synonyns of w1
            try:
                syns = wn.synsets(right_context_tokens[0])
                i = 0
                for each in syns[0:2]:
                    if each:
                        features[instance_id]['w1_syn_' + str(i)] = each.name()
                        i += 1
            except IndexError:
                i = 0
                for each in range(0,3):
                    features[instance_id]['w1_syn_' + str(i)] = ''
                    i += 1  
            
            #combine contexts and head, perform sentence segmentation, and tag parts of speech

#             head_index = len(nltk.word_tokenize(lcontext)) #for finding head later, check using print statement
#             full_context = lcontext.strip() + ' ' + head.strip() + ' ' + rcontext.strip()
#             sentences = sent_detector.tokenize(full_context)
            POStokens = left_context_all[-6:] + [head] + right_context_all[0:5]
            tagged = tagger.tag(POStokens)

            #postags
        
            i = -3
            for tag in tagged[len(left_context_all)-3:len(left_context_all)+3]:
    
                features[instance_id]['pos' + str(i)] = tag[1]
                i += 1        
            
            
            # relevance score
#             
#             window_size=5
#    
#             left =left_context_tokens[(-1*window_size):]
#             right= right_context_tokens[0:window_size]
#    
#    
#             unique_context= set(left+right)
#    
#             score=[]
#    
#    
#             for item in unique_context:
#        
#                 nsc_count=nsc[i[4]][item]
#                 nsc_countbar=nc[item]-nsc_count
#        
#                if nsc_countbar==0:
#                    score.append((10000,item))
#        
#                else:
#                    score.append((float(math.log(float(nsc_count))-math.log(float(nsc_countbar))),item))
# 
#            score=sorted(score)
#         
#            # print score
#    
#            if len(score)>=1:
#                features[i[0]]['relevance_1']=score[-1][1]
# 
#            if len(score)>=2:
# 
#                features[i[0]]['relevance_2']=score[-2][1]
#    
#            if len(score)>=3:
#                features[i[0]]['relevance_3']=score[-3][1]
#    
#            if len(score)>=4:
#                features[i[0]]['relevance_4']=score[-4][1]
#         
        
            
    #SPANISH LOOP
        
    if sys.argv[-1] == 'Spanish':
    
        # print 'SPANISH'
        for inst in data:
        
            instance_id = inst[0]
            lcontext = inst[1]
            head = inst[2]
            rcontext = inst[3]
            sense_id = inst[4]
        
                
            features[instance_id] = {}
        
        
            if instance_id and sense_id: 
                labels[instance_id] = sense_id
        
            left_context_all = nltk.word_tokenize(lcontext)
            left_context_tokens = [token for token in left_context_all if not token in string.punctuation] #and not token in stopwords.words('spanish')
            left_context_tokens = left_context_tokens[-window_size:]
            
            right_context_all = nltk.word_tokenize(rcontext)
            right_context_tokens = [token for token in right_context_all if not token in string.punctuation] # and not token in stopwords.words('spanish')
            right_context_tokens = right_context_tokens[0:window_size]
        
            context = left_context_tokens + right_context_tokens
          
    #         instance_windowtokens[instance_id] = context
        
            #surrounding words, no stopwords or punctuation
        
            for i in range(-3,0):
                try: features[instance_id]['w_' + str(-i)] = left_context_tokens[i]
                except IndexError: features[instance_id]['w_' + str(-i)] = ''
        
            for i in range(0,3):
                try: features[instance_id]['w' + str(i)] = right_context_tokens[i]
                except IndexError: features[instance_id]['w' + str(i)] = ''
        
            #collocations
        
            left_tokens_all = [token for token in left_context_all if not token in string.punctuation]
            right_tokens_all = [token for token in right_context_all if not token in string.punctuation]
        
            try: features[instance_id]['c-3-2-1'] = left_tokens_all[-3] + '_' + left_tokens_all[-2] + '_' + left_tokens_all[-1]
            except IndexError: features[instance_id]['c-3-2-1'] = ''
        
            try: features[instance_id]['c-2-2'] = left_tokens_all[-2]
            except IndexError: features[instance_id]['c-2-2'] = ''
        
            try: features[instance_id]['c-2-1'] = left_tokens_all[-2] + '_' + left_tokens_all[-1]
            except IndexError: features[instance_id]['c-2-1'] = ''
        
            try: features[instance_id]['c-1-1'] = left_tokens_all[-1]
            except IndexError: features[instance_id]['c-1-1'] = ''
        
            try: features[instance_id]['c-2_1'] = left_tokens_all[-2] + '_' + left_tokens_all[-1] + right_tokens_all[0]
            except IndexError: features[instance_id]['c-2_1'] = ''
        
            try: features[instance_id]['c11'] = right_tokens_all[0]
            except IndexError: features[instance_id]['c11'] = ''
        
            try: features[instance_id]['c12'] = right_tokens_all[0] + '_' + right_tokens_all[1]
            except IndexError: features[instance_id]['c12'] = ''
        
            try: features[instance_id]['c22'] = right_tokens_all[1]
            except IndexError: features[instance_id]['c22'] = ''
        
            try: features[instance_id]['c-1_1'] = left_tokens_all[-1] + '_' + right_tokens_all[0]
            except IndexError: features[instance_id]['c-1_1'] = ''
        
            try: features[instance_id]['c-1_2'] = left_tokens_all[-1] + '_' + right_tokens_all[0] + '_' + right_tokens_all[1]
            except IndexError: features[instance_id]['c-1_2'] = ''
        
            try: features[instance_id]['c123'] = right_tokens_all[0] + '_' + right_tokens_all[1] + '_' + right_tokens_all[2]
            except IndexError: features[instance_id]['c123'] = ''

            #synset - top 3 synonyns of head
            syns = wn.synsets(instance_id.split('.')[0], instance_id.split('.')[1], lang = 'spa')
            i = 0
            for each in syns[0:2]:
                if each:
                    features[instance_id]['head_syn_' + str(i)] = each.name()
                    i += 1
    #         
    #         #synset - top 3 synonyns of w-2
    #         try:
    #           syns = wn.synsets(left_context_tokens[-2])
    #           i = 0
    #           for each in syns[0:2]:
    #               if each:
    #                   features[instance_id]['w-2_syn_' + str(i)] = each.name()
    #                   i += 1
    #         except IndexError:
    #             i = 0
    #             for each in range(0,3):
    #               features[instance_id]['w-2_syn_' + str(i)] = ''
    #               i += 1
        
            #synset - top 3 synonyns of w-1
            try:
                syns = wn.synsets(left_context_tokens[-1], lang = 'spa')
                i = 0
                for each in syns[0:2]:
                    if each:
                        features[instance_id]['w-1_syn_' + str(i)] = each.name()
                        i += 1
            except IndexError:
                i = 0
                for each in range(0,3):
                    features[instance_id]['w-1_syn_' + str(i)] = ''
                    i += 1
        
        
        
            #synset - top 3 synonyns of w1
            try:
                syns = wn.synsets(right_context_tokens[0], lang = 'spa')
                i = 0
                for each in syns[0:2]:
                    if each:
                        features[instance_id]['w1_syn_' + str(i)] = each.name()
                        i += 1
            except IndexError:
                i = 0
                for each in range(0,3):
                    features[instance_id]['w1_syn_' + str(i)] = ''
                    i += 1          




    #CATALAN LOOP
        
    if sys.argv[-1] == 'Catalan':
    
        # print 'CATALAN'
        for inst in data:
        
            instance_id = inst[0]
            lcontext = inst[1]
            head = inst[2]
            rcontext = inst[3]
            sense_id = inst[4]
        
                
            features[instance_id] = {}
        
        
            if instance_id and sense_id: 
                labels[instance_id] = sense_id
        
            left_context_all = nltk.word_tokenize(lcontext)
            left_context_tokens = [token for token in left_context_all if not token in string.punctuation] #and not token in stopwords.words('spanish')
            left_context_tokens = left_context_tokens[-window_size:]
            
            right_context_all = nltk.word_tokenize(rcontext)
            right_context_tokens = [token for token in right_context_all if not token in string.punctuation] # and not token in stopwords.words('spanish')
            right_context_tokens = right_context_tokens[0:window_size]
        
            context = left_context_tokens + right_context_tokens
          
    #         instance_windowtokens[instance_id] = context
        
            #surrounding words, no stopwords or punctuation
        
            for i in range(-3,0):
                try: features[instance_id]['w_' + str(-i)] = left_context_tokens[i]
                except IndexError: features[instance_id]['w_' + str(-i)] = ''
        
            for i in range(0,3):
                try: features[instance_id]['w' + str(i)] = right_context_tokens[i]
                except IndexError: features[instance_id]['w' + str(i)] = ''
        
            #collocations
        
            left_tokens_all = [token for token in left_context_all if not token in string.punctuation]
            right_tokens_all = [token for token in right_context_all if not token in string.punctuation]
        
            try: features[instance_id]['c-3-2-1'] = left_tokens_all[-3] + '_' + left_tokens_all[-2] + '_' + left_tokens_all[-1]
            except IndexError: features[instance_id]['c-3-2-1'] = ''
        
            try: features[instance_id]['c-2-2'] = left_tokens_all[-2]
            except IndexError: features[instance_id]['c-2-2'] = ''
        
            try: features[instance_id]['c-2-1'] = left_tokens_all[-2] + '_' + left_tokens_all[-1]
            except IndexError: features[instance_id]['c-2-1'] = ''
        
            try: features[instance_id]['c-1-1'] = left_tokens_all[-1]
            except IndexError: features[instance_id]['c-1-1'] = ''
        
            try: features[instance_id]['c-2_1'] = left_tokens_all[-2] + '_' + left_tokens_all[-1] + right_tokens_all[0]
            except IndexError: features[instance_id]['c-2_1'] = ''
        
            try: features[instance_id]['c11'] = right_tokens_all[0]
            except IndexError: features[instance_id]['c11'] = ''
        
            try: features[instance_id]['c12'] = right_tokens_all[0] + '_' + right_tokens_all[1]
            except IndexError: features[instance_id]['c12'] = ''
        
            try: features[instance_id]['c22'] = right_tokens_all[1]
            except IndexError: features[instance_id]['c22'] = ''
        
            try: features[instance_id]['c-1_1'] = left_tokens_all[-1] + '_' + right_tokens_all[0]
            except IndexError: features[instance_id]['c-1_1'] = ''
        
            try: features[instance_id]['c-1_2'] = left_tokens_all[-1] + '_' + right_tokens_all[0] + '_' + right_tokens_all[1]
            except IndexError: features[instance_id]['c-1_2'] = ''
        
            try: features[instance_id]['c123'] = right_tokens_all[0] + '_' + right_tokens_all[1] + '_' + right_tokens_all[2]
            except IndexError: features[instance_id]['c123'] = ''

            #synset - top 3 synonyns of head
            syns = wn.synsets(instance_id.split('.')[0], instance_id.split('.')[1], lang = 'cat')
            i = 0
            for each in syns[0:2]:
                if each:
                    features[instance_id]['head_syn_' + str(i)] = each.name()
                    i += 1
    #         
    #         #synset - top 3 synonyns of w-2
    #         try:
    #           syns = wn.synsets(left_context_tokens[-2])
    #           i = 0
    #           for each in syns[0:2]:
    #               if each:
    #                   features[instance_id]['w-2_syn_' + str(i)] = each.name()
    #                   i += 1
    #         except IndexError:
    #             i = 0
    #             for each in range(0,3):
    #               features[instance_id]['w-2_syn_' + str(i)] = ''
    #               i += 1
        
            #synset - top 3 synonyns of w-1
            try:
                syns = wn.synsets(left_context_tokens[-1], lang = 'cat')
                i = 0
                for each in syns[0:2]:
                    if each:
                        features[instance_id]['w-1_syn_' + str(i)] = each.name()
                        i += 1
            except IndexError:
                i = 0
                for each in range(0,3):
                    features[instance_id]['w-1_syn_' + str(i)] = ''
                    i += 1
        
        
        
            #synset - top 3 synonyns of w1
            try:
                syns = wn.synsets(right_context_tokens[0], lang = 'cat')
                i = 0
                for each in syns[0:2]:
                    if each:
                        features[instance_id]['w1_syn_' + str(i)] = each.name()
                        i += 1
            except IndexError:
                i = 0
                for each in range(0,3):
                    features[instance_id]['w1_syn_' + str(i)] = ''
                    i += 1          
    







    
    return features, labels

# implemented for you
def vectorize(train_features,test_features):
    '''
    convert set of features to vector representation
    :param train_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :param test_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :return: X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
            X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    '''
    X_train = {}
    X_test = {}

    vec = DictVectorizer()
    vec.fit(train_features.values())
    for instance_id in train_features:
        X_train[instance_id] = vec.transform(train_features[instance_id]).toarray()[0]

    for instance_id in test_features:
        X_test[instance_id] = vec.transform(test_features[instance_id]).toarray()[0]

    return X_train, X_test

#B.1.e
def feature_selection(X_train,X_test,y_train):
    '''
    Try to select best features using good feature selection methods (chi-square or PMI)
    or simply you can return train, test if you want to select all features
    :param X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }
    :return:
    '''

    

    # implement your code here
#     
#     for instance_id, vector in X_train.iteritems():
#         answers.append(y_train[instance_id])
#         vectors.append(vector)
#     ch2 = SelectKBest(chi2,k=2)
#     X_train_new = ch2.fit_transform(vectors,answers)
#     Xtest_new = []
#     for instance_id,vector in X_test.iteritems():
#         Xtest.append(vector)
#     X_test_new = ch2.transform(Xtest)
    #return X_train_new, X_test_new
    # or return all feature (no feature selection):
    return X_train, X_test

# B.2
def classify(X_train, X_test, y_train):
    '''
    Train the best classifier on (X_train, and y_train) then predict X_test labels

    :param X_train: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param X_test: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }

    :return: results: a list of tuples (instance_id, label) where labels are predicted by the best classifier
    '''
    
    # implement your code here
       
    xtrain = []
    ytrain = []
    xtest = []
    results = []
    
    for instance_id, value in X_train.items():
        xtrain.append(value)
        ytrain.append(y_train[instance_id])
 
    for instance_id, value in X_test.items():
        xtest.append(value)
   
    svm_clf = svm.LinearSVC()
    svm_clf.fit(xtrain, ytrain)
    svm_labels = svm_clf.predict(xtest)
    i=0
    for key in X_test.keys():
        results.append((key, svm_labels[i]))
        i+=1
    
    
    return results

# run part B
def run(train, test, language, answer):
    results = {}

    for lexelt in train:

        train_features, y_train = extract_features(train[lexelt])
        test_features, _ = extract_features(test[lexelt])

        X_train, X_test = vectorize(train_features,test_features)
        X_train_new, X_test_new = feature_selection(X_train, X_test,y_train)
        results[lexelt] = classify(X_train_new, X_test_new,y_train)

    A.print_results(results, answer)