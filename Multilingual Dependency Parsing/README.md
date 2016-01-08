#Multilingual Dependency Parsing

The scripts included in this repository implement a dependency parsing algorithm in English, Danish, and Swedish. The parser implements a version of Joakim Nivre's acr-eager transition-based dependency parser.

This parser uses data from the CoNLL-X shared task on multilingual dependency parsing, specifically the English, Swedish, and Danish data sets.

FEatureextractor.py is used to extract features from the data, which are then used to train the SVM classifier "oracle" used to parse dependencies.

For more information on Nivre's acr-eager transition-based dependency parser, see the paper "A Dynamic Oracle for Arc-Eager Dependency Parsing" here: http://aclweb.org/anthology/C/C12/C12-1059.pdf
