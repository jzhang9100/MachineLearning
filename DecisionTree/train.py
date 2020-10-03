#!bin/usr/python3
from datasets import datasets
from dtree import dtree
from collections import Counter

import numpy as np

d = datasets()

SPAMDATA = 'data/spam/spam.data'
(spam_x, spam_y) = d.load_spam(SPAMDATA)

spam_tree = dtree(spam_x, spam_y)
spam_tree.print_data()

spam_tree.fit(10)

VOLCANODATA = 'data/volcanoes/volcanoes.data'
(volcano_x, volcano_y) = d.load_volcanoes(VOLCANODATA)

# volcano_tree = dtree(volcano_x, volcano_y)
# volcano_tree.print_data()
