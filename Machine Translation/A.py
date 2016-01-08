import nltk
from nltk.align import IBMModel1, IBMModel2
from nltk.corpus import comtrans
from nltk.align import AlignedSent

# TODO: Initialize IBM Model 1 and return the model.
def create_ibm1(aligned_sents):
    # initialize instance of IBM model 1 using 10 iterations of EM, training on aligned_sents
    imb1 = IBMModel1(aligned_sents, 10)
    
    return imb1

# TODO: Initialize IBM Model 2 and return the model.
def create_ibm2(aligned_sents):
    
    ibm2 = IBMModel2(aligned_sents, 10)
    
    return ibm2
    
    
# TODO: Compute the average AER for the first n sentences
#       in aligned_sents using model. Return the average AER.
def compute_avg_aer(aligned_sents, model, n):
    
    results = []
    
    for x in aligned_sents[0:n]:
        modelAlign = model.align(x)
        AER = modelAlign.alignment_error_rate(x)
        print str(x) + ' ' + str(AER)
        results.append(AER)
    
    return float(sum(results))/float(len(results))

#     pass    
    
    
    
# TODO: Computes the alignments for the first 20 sentences in
#       aligned_sents and saves the sentences and their alignments
#       to file_name. Use the format specified in the assignment.
def save_model_output(aligned_sents, model, file_name):
    #save model's predicted alignments to "ibm1.txt"
    
    file = open(file_name, mode = 'w')
    
    for sent in aligned_sents[0:20]:
        newsent = model.align(sent)
        file.write(str(newsent.words) + '\n')
        file.write(str(newsent.mots) + '\n')
        file.write(str(newsent.alignment) + '\n')
        file.write('\n')
    
    file.close()
        

def main(aligned_sents):
    ibm1 = create_ibm1(aligned_sents)
    save_model_output(aligned_sents, ibm1, "ibm1.txt")
    avg_aer = compute_avg_aer(aligned_sents, ibm1, 50)

    print ('IBM Model 1')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))

    ibm2 = create_ibm2(aligned_sents)
    save_model_output(aligned_sents, ibm2, "ibm2.txt")
    avg_aer = compute_avg_aer(aligned_sents, ibm2, 50)
    
    print ('IBM Model 2')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
