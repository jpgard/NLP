from main import replace_accented
from sklearn import svm
from sklearn import neighbors
import nltk
import string


# don't change the window size
window_size = 10

# A.1
def build_s(data):
    '''
    Compute the context vector for each lexelt
    :param data: dict with the following structure:
        {
            lexelt: [(instance_id, left_context, head, right_context, sense_id), ...],
            ...
        }
    :return: dict s with the following structure:
        {
            lexelt: [w1,w2,w3, ...],
            ...
        }

    '''
    s = {}
    
    # implement your code here
    
    for key,value in data.items():
        
        output = []
        
        for val in value:
        
            
            left_context_tokens = nltk.word_tokenize(val[1])
            left_context_tokens = [token for token in left_context_tokens if not token in string.punctuation]
            left_context_tokens = left_context_tokens[-10:]
            
            right_context_tokens = nltk.word_tokenize(val[3])
            right_context_tokens = [token for token in right_context_tokens if not token in string.punctuation]
            right_context_tokens = right_context_tokens[0:9]
            
            
            for x in left_context_tokens + right_context_tokens:
                if not key in s:
                    s[key] = []
                    s[key].append(x)
                    continue
                if not x in s[key]:
                    s[key].append(x)
    
    return s


# A.1
def vectorize(data, s):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
            [(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :param s: list of words (features) for a given lexelt: [w1,w2,w3, ...]
    :return: vectors: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }

    '''
    vectors = {}
    labels = {}

 # implement your code here
  
#     #vectors

    for inst in data:

        #add entry to 'labels' for each instance/sense pair
        
        instance_id = inst[0]
        sense_id = inst[4]
        if instance_id and sense_id: labels[instance_id] = sense_id
        
        
        #compute vectors and add entry to vectors for each instance/w_counts pair
        
        w_counts = []
        
        left_context_tokens = nltk.word_tokenize(inst[1])
        left_context_tokens = [token for token in left_context_tokens if not token in string.punctuation]
        left_context_tokens = left_context_tokens[-10:]
        
        right_context_tokens = nltk.word_tokenize(inst[3])
        right_context_tokens = [token for token in right_context_tokens if not token in string.punctuation]
        right_context_tokens = right_context_tokens[0:9]
        left_right_context = left_context_tokens + right_context_tokens
        
        
        for word in s:
            count = 0
            for each in left_right_context:
                if each == word:
                    count += 1
            w_counts.append(count)
        
        vectors[instance_id] = w_counts
            
    # print vectors.items()
    # print labels.items()
    return vectors, labels


# A.2
def classify(X_train, X_test, y_train):
    '''
    Train two classifiers on (X_train, and y_train) then predict X_test labels

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

    :return: svm_results: a list of tuples (instance_id, label) where labels are predicted by LinearSVC
             knn_results: a list of tuples (instance_id, label) where labels are predicted by KNeighborsClassifier
    '''

    svm_results = []
    knn_results = []

    
    svm_clf = svm.LinearSVC()
    knn_clf = neighbors.KNeighborsClassifier()
    
    # implement your code here
    
    svm_clf.fit(X_train.values(), y_train.values())
    knn_clf.fit(X_train.values(), y_train.values())
    
    svm_predictions = svm_clf.predict(X_test.values())
    knn_predictions = knn_clf.predict(X_test.values())
    
    for j in range(len(X_test.items())):
        svm_results.append((X_test.items()[j][0], svm_predictions[j]))
        knn_results.append((X_test.items()[j][0], knn_predictions[j]))
    
    
    return svm_results, knn_results

# A.3, A.4 output
def print_results(results ,output_file):
    '''

    :param results: A dictionary with key = lexelt and value = a list of tuples (instance_id, label)
    :param output_file: file to write output

    '''

    # implement your code here
    # don't forget to remove the accent of characters using main.replace_accented(input_str)
    # you should sort results alphabetically by lexelt_item, then on 
    # instance_id before printing
    
    out = open(output_file, 'w')
    
    for lexelt, tuples in sorted(results.items()):
        x = replace_accented(lexelt)
        for instance in sorted(tuples):
            y = replace_accented(instance[0])
            z = instance[1]
            out.write(x +' ' + y +' '+ z + '\n')
    
    out.close()

# run part A
def run(train, test, language, knn_file, svm_file):
    s = build_s(train)
    svm_results = {}
    knn_results = {}
    for lexelt in s:
        X_train, y_train = vectorize(train[lexelt], s[lexelt])
        X_test, _ = vectorize(test[lexelt], s[lexelt])
        svm_results[lexelt], knn_results[lexelt] = classify(X_train, X_test, y_train)

    print_results(svm_results, svm_file)
    print_results(knn_results, knn_file)



