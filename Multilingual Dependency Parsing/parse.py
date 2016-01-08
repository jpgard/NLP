
from providedcode import dataset
from providedcode.transitionparser import TransitionParser
from providedcode.evaluate import DependencyEvaluator
from featureextractor import FeatureExtractor
from transition import Transition
import sys
from providedcode.dependencygraph import DependencyGraph


if __name__ == '__main__':
    
    englishfile = sys.stdin
    tp = TransitionParser.load('english.model')
    
    
    while True:
        x = englishfile.readline()
        x = x.rstrip()
        if not x: break
        sentence = DependencyGraph.from_sentence(x)
        parsed = tp.parse([sentence])
        for i in parsed:
            print i.to_conll(10).encode('utf-8')
    
    
    # while True:
#         x = englishfile.readline()
#         x = x.rstrip()
#         if not x: break
#         sentence = DependencyGraph.from_sentence(x)
#         parsed = tp.parse([sentence])
#         for i in parsed:
#             print i.to_conll(10).encode('utf-8')
