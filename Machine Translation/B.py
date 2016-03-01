import A, nltk, sys
from collections import defaultdict
from nltk.align import AlignedSent

class BerkeleyAligner():

    def __init__(self, align_sents, num_iter):
        self.t, self.q = self.train(align_sents, num_iter)

    # TODO: Computes the alignments for align_sent, using this model's parameters. Return
    #       an AlignedSent object, with the sentence pair and the alignments computed.
    def align(self, align_sent):
       
        t = self.t
        q = self.q
        alignment = []
        best_alignment = []
        
        m = len(align_sent.mots)
        l = len(align_sent.words)
        
        for j, en_word in enumerate(align_sent.words):
            
            # Initialize the maximum probability with Null token
            max_align_prob = -sys.maxint
            
            for i, g_word in enumerate(align_sent.mots):
                # Find out the maximum probability
                max_align_prob = max(max_align_prob,
                    (t[(en_word, g_word)]*q[(i,j,m,l)], i), (t[(g_word,en_word)]*q[(j, i,l,m)], i))

            alignment.append((j, max_align_prob[1]))
             
        return AlignedSent(align_sent.words, align_sent.mots, alignment)
        
    
    # TODO: Implement the EM algorithm. num_iters is the number of iterations. Returns the 
    # translation and distortion parameters as a tuple.
    def train(self, aligned_sents, num_iters):
        t = defaultdict(float)
        q = defaultdict(float)   
        
        e_targetwords = defaultdict(set)
        g_targetwords = defaultdict(set)
        
        #first pass through aligned_sents to compile lists of potential target words and vocabulary for e/g
        
        for sent in aligned_sents:
            e_sent = sent.words
            g_sent = sent.mots
            
            for e in e_sent: #english sentence: add each word to counts; add targetwords to relevant entry   
                g_targetwords[e].update(g_sent)
                
            for g in g_sent: #german sentence: add each word to counts; add targetwords to relevant entry 
                e_targetwords[g].update(e_sent)
                
        for sentence in aligned_sents:
            for f in sentence.words:
                for e in sentence.mots:
                    t[(f,e)] = float(1)/float(len(e_targetwords[e]))
                    t[(e,f)] = float(1)/float(len(g_targetwords[f]))
            m = len(sentence.mots)
            l = len(sentence.words)
            
            for i in range(m):
                for j in range(l):
                    q[(j,i,l,m)] = float(1)/float(l+1)
                    q[(i,j,m,l)] = float(1)/float(m+1)
        
        #initialization complete.
        
        for S in range(num_iters):
            c = defaultdict(float)
            for sentence in aligned_sents:
                f = sentence.mots
                e = sentence.words
                m = len(sentence.mots)
                l = len(sentence.words)
                
                for i in range(m):
                
                    for j in range(l):
                
                        d_1 = q[(j,i,l,m)]*t[(f[i], e[j])]
                        
                        sum_1 = 0
                        for x in range(l):
                            sum_1 += q[(x,i,l,m)]*t[(f[i], e[x])]
                        d_1 = float(d_1)/float(sum_1)
                        
                        d_2 = q[(i,j,m,l)]*t[(e[j], f[i])]
                        sum_2 = 0
                        for x in range(l):
                            sum_2 += q[(i,x,m,l)]*t[(e[x], f[i])]
                        d_2 = float(d_2)/float(sum_2)
                        
                        d = float(d_1 + d_2)/float(2)
                        c[(e[j], f[i])] += d
                        c[(f[i], e[j])] += d
                        c[e[j]] += d 
                        c[f[i]] += d
                        c[(j,i,l,m)] += d
                        c[(i,j,m,l)] += d
                        c[(i,l,m)] += d
                        c[(j,m,l)] += d
        
            for sentence in aligned_sents:
                for e in sentence.words:
                    for f in sentence.mots:
                        t[(f,e)] = (c[(e,f)]/c[e])
                        t[(e,f)] = (c[(f,e)]/c[f])
                m = len(sentence.mots)
                l = len(sentence.words)
                for i in range(m):
                    for j in range(l):
                        q[(j,i,l,m)] = c[(j,i,l,m)]/c[(i,l,m)]
                        q[(i,j,m,l)] = c[(i,j,m,l)]/c[(j,m,l)]
                                               
        return (t,q)

def main(aligned_sents):
    ba = BerkeleyAligner(aligned_sents, 10)
    A.save_model_output(aligned_sents, ba, "ba.txt")
    avg_aer = A.compute_avg_aer(aligned_sents, ba, 50)
    
    print ('Berkeley Aligner')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
