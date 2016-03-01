from nltk.compat import python_2_unicode_compatible

printed = False

@python_2_unicode_compatible
class FeatureExtractor(object):
    @staticmethod
    def _check_informative(feat, underscore_is_informative=False):
        """
        Check whether a feature is informative
        """

        if feat is None:
            return False

        if feat == '':
            return False

        if not underscore_is_informative and feat == '_':
            return False

        return True

    @staticmethod
    def find_left_right_dependencies(idx, arcs):
        left_most = 1000000
        right_most = -1
        dep_left_most = ''
        dep_right_most = ''
        for (wi, r, wj) in arcs:
            if wi == idx:
                if (wj > wi) and (wj > right_most):
                    right_most = wj
                    dep_right_most = r
                if (wj < wi) and (wj < left_most):
                    left_most = wj
                    dep_left_most = r
        return dep_left_most, dep_right_most

    
    @staticmethod
    def extract_features(tokens, buffer, stack, arcs):
        """
        This function returns a list of string features for the classifier

        :param tokens: nodes in the dependency graph
        :param stack: partially processed words
        :param buffer: remaining input words
        :param arcs: partially built dependency tree

        :return: list(str)
        """

        """
        Implementation of features below based on some standard features as
        described in Table 3.2 on page 31 of Dependency Parsing by Kubler,
        McDonald, and Nivre

        [http://books.google.com/books/about/Dependency_Parsing.html?id=k3iiup7HB9UC]
        """

        result = []


        global printed
        if not printed:
            # print("This is not a very good feature extractor!")
            printed = True

        if stack:
            stack_idx0 = stack[-1]
            token = tokens[stack_idx0]
            
            if FeatureExtractor._check_informative(token['word'], True):
                result.append('STK_0_FORM_' + token['word'])

            if 'feats' in token and FeatureExtractor._check_informative(token['feats']):
                feats = token['feats'].split("|")
                for feat in feats:
                    result.append('STK_0_FEATS_' + feat)

            # Left most, right most dependency of stack[0]
            dep_left_most, dep_right_most = FeatureExtractor.find_left_right_dependencies(stack_idx0, arcs)

            if FeatureExtractor._check_informative(dep_left_most):
                result.append('STK_0_LDEP_' + dep_left_most)
            if FeatureExtractor._check_informative(dep_right_most):
                result.append('STK_0_RDEP_' + dep_right_most)
            
            # Fine-grained POS tag of stack[0]
            if 'tag' in token and FeatureExtractor._check_informative(token['tag']):
                postag = token['tag']
                result.append('STK_0_POSTAG_' + postag)
                
            # STK_0_LEMMA_ [lemma or base form of word on top of stack] 
            if 'lemma' in token and FeatureExtractor._check_informative(token['lemma']):
                lemma = token['lemma']
                result.append('STK_0_LEMMA_' + lemma)
        
        #form and POS and lemma of second word in stack, if present
        
        if len(stack) > 1:
            stack_idx1 = stack[-2]
            token = tokens[stack_idx1]
            
            if 'tag' in token and FeatureExtractor._check_informative(token['tag']):
                tag = token['tag']
                result.append('STK_1_POSTAG_' + tag)
#            
            if 'lemma' in token and FeatureExtractor._check_informative(token['lemma']):
                lemma = token['lemma']
                result.append('STK_1_LEMMA_' + lemma)
               
        if len(stack) > 2:
            stack_idx2 = stack[-3]
            token = tokens[stack_idx2]
            
            if 'tag' in token and FeatureExtractor._check_informative(token['tag']):
                tag = token['tag']
                result.append('STK_2_POSTAG_' + tag)
                
        if len(stack) > 3:
            stack_idx3 = stack[-4]
            token = tokens[stack_idx3]
            
            if 'tag' in token and FeatureExtractor._check_informative(token['tag']):
                tag = token['tag']
                result.append('STK_3_POSTAG_' + tag)
#        
        if buffer:
            buffer_idx0 = buffer[0]
            token = tokens[buffer_idx0]
            if FeatureExtractor._check_informative(token['word'], True):
                result.append('BUF_0_FORM_' + token['word'])

            if 'feats' in token and FeatureExtractor._check_informative(token['feats']):
                feats = token['feats'].split("|")
                for feat in feats:
                    result.append('BUF_0_FEATS_' + feat)

            dep_left_most, dep_right_most = FeatureExtractor.find_left_right_dependencies(buffer_idx0, arcs)

            if FeatureExtractor._check_informative(dep_left_most):
                result.append('BUF_0_LDEP_' + dep_left_most)
            if FeatureExtractor._check_informative(dep_right_most):
                result.append('BUF_0_RDEP_' + dep_right_most)
            
            # Fine-grained POS of buffer[0]
            if 'tag' in token and FeatureExtractor._check_informative(token['tag']):
                postag = token['tag']
                result.append('BUF_0_POSTAG_' + postag)
                
            #coarse POS of buffer[0]
#             if 'ctag' in token and FeatureExtractor._check_informative(token['ctag']):
#                 cpostag = token['ctag']
#                 result.append('BUF_0_CPOSTAG_' + cpostag)
            
            # BUF_0_LEMMA_ [lemma or base form of word on top of buffer]
            if 'lemma' in token and FeatureExtractor._check_informative(token['lemma']):
                lemma = token['lemma']
                result.append('BUF_0_LEMMA_' + lemma)
        
        #form and POS and lemma of second word in stack, if present
                
        if len(buffer) > 1:
            buffer_idx1 = buffer[1]
            token = tokens[buffer_idx1]
            
#             if 'word' in token and FeatureExtractor._check_informative(token['word']):
#                 form = token['word']
#                 result.append('BUF_1_FORM_' + form)
            
            if 'tag' in token and FeatureExtractor._check_informative(token['tag']):
                tag = token['tag']
                result.append('BUF_1_POSTAG_' + tag)
#            
            if 'lemma' in token and FeatureExtractor._check_informative(token['lemma']):
                lemma = token['lemma'] 
                result.append('BUF_1_LEMMA_' + lemma)
        
        if len(buffer) > 2:
            buffer_idx2 = buffer[2]
            token = tokens[buffer_idx2]
            
            if 'tag' in token and FeatureExtractor._check_informative(token['tag']):
                tag = token['tag']
                result.append('BUF_2_POSTAG_' + tag)
            
            if 'lemma' in token and FeatureExtractor._check_informative(token['lemma']):
                lemma = token['lemma'] 
                result.append('BUF_2_LEMMA_' + lemma)
            
        
        if len(buffer) > 3:
            buffer_idx3 = buffer[3]
            token = tokens[buffer_idx3]   
            
            if 'tag' in token and FeatureExtractor._check_informative(token['tag']):
                tag = token['tag']
                result.append('BUF_3_POSTAG_' + tag)
                
            if 'lemma' in token and FeatureExtractor._check_informative(token['lemma']):
                lemma = token['lemma'] 
                result.append('BUF_3_LEMMA_' + lemma)
        
        #distance between words
#                
        if stack and buffer:
            buffer_idx0 = buffer[0]
            buffer_token = tokens[buffer_idx0]
            stack_idx0 = stack[-1]
            stack_token = tokens[stack_idx0]
            
            if 'address' in stack_token and 'address' in buffer_token and FeatureExtractor._check_informative(buffer_token['address']) and FeatureExtractor._check_informative(stack_token['address']):
                buffer_address = int(buffer_token['address'])
                stack_address = int(stack_token['address'])
                distance = stack_address - buffer_address
                result.append('STK0_BUF0_DISTANCE_' + str(distance))
             

        return result
