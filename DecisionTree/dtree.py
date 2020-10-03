#!bin/usr/python3
#Jack Zhang

import math
import numpy as np
import pandas as pd
from collections import defaultdict

class dTreeNode:
    def __init__(self, split_index=0, , split_val=0, left=None, right=None):
        self.split_index = split_index
        self.split_val = split_val
        self.left = left
        self.right = right

class dtree:
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
        self.num_ones = 0
        self.hy = 0
        for label in self.y:
            if label == 1:
                self.num_ones += 1

    def print_data(self):
        print(self.x, self.y)

    # let labels be an array of labels valued at either 1 or 0
    def entropy_calculation_of_y(self, labels):
        # First get the proportion of 1s in the labels
        tot = len(labels)
        unique, count = np.unique(labels, return_counts=True)
        d = dict(zip(unique, count))
        self.num_ones = d[1]
        print(self.num_ones, tot)

        # print(self.num_ones)
        factor = (self.num_ones/tot)

        return -factor*math.log2(factor) - (1-factor)*math.log2(1-factor)

    # let col be an array giving the values for a certain column of data
    def entropy_calculation(self, col, labels):
        col_entries = defaultdict(int)
        for i in range(len(col)):
            val = str(col[i])
            col_entries[val] += 1
            if labels[i] == 1:
                col_entries[val + '_1'] += 1
        num_unique_col_entries = len(col)
        entropy = 0.0
        max_entropy = 0.0
        for key in list(col_entries):
            if '_' not in key:
                # Probability of Y|Feature
                py = col_entries[key]/num_unique_col_entries
                if col_entries[key+'_1'] == 0:
                    hy = 0
                else:
                    hy = math.log2(col_entries[key + '_1']/col_entries[key])
                entropy += -py * hy
                if entropy > max_entropy:
                    max_entropy = entropy
        return entropy

    def max_entropy(self, col, labels):
        col_entries = defaultdict(int)
        for i in range(len(col)):
            val = str(col[i])
            col_entries[val] += 1
            if labels[i] == 1:
                col_entries[val + '_1'] += 1
        num_unique_col_entries = len(col)
        temp_entropy = 0.0
        max_entropy = 0.0
        for key in list(col_entries):
            if '_' not in key:
                # Probability of Y|Feature
                py = col_entries[key]/num_unique_col_entries
                if col_entries[key + '_1'] == 0:
                    hy = 0
                else:
                    hy = math.log2(col_entries[key + '_1'] / col_entries[key])
                temp_entropy = -py * hy
                if temp_entropy > max_entropy:
                    max_entropy = temp_entropy
        return max_entropy

    def gain_ratio(self, col, labels, hy):
        return self.information_gain(col, labels)/hy

    def information_gain(self, col, labels, hy):
        return hy - self.entropy_calculation(col, labels)

    #N Cross Fold Validation
    def fit(self, n):
        split_indices = []
        k = len(self.x)/n
        i = 0
        while i+k < len(self.x):
            split_indices.append([int(i), int(i+k)])
            i += k

        for j in range(len(split_indices)):
            test_indices = split_indices[j]
            train_indices = [split_indices[x] for x in range(len(split_indices)) if x != j]

            #print(test_indices, train_indices)
            test_x, test_y = self.x[test_indices[0]:test_indices[1]], self.y[test_indices[0]:test_indices[1]]

            train_x, train_y = [], []
            for inx in train_indices:
                train_x.append(pd.DataFrame(self.x[inx[0]:inx[1]]))
                train_y.append(pd.DataFrame(self.y[inx[0]:inx[1]]))

            x_train = pd.concat(train_x)
            del x_train['index']
            y_train = pd.concat(train_y)

            hy = self.entropy_calculation_of_y(np.array(y_train))
            self.dTreeRoot = self.train_step(None, x_train, y_train, set(), hy)
            break



    def train_step(self, node, data, labels, seen, hy, max_depth=None):
        x = np.array(data)
        y = np.array(labels)
        xtrans = np.transpose(x)
        if len(seen) != len(xtrans):
            mx = float('-inf')
            split_index = -1
            for f in range(len(xtrans)):
                if f not in seen:
                    feature = xtrans[f]
                    print(feature.shape, y.shape)
                    ig = self.information_gain(feature, y, hy)
                    print(ig)
                    if ig > mx:
                        mx = ig
                        split_index = f

            seen.add(split_index)
            split_val = self.max_entropy(xtrans[split_index], y)

            if not node:
                root = dTreeNode(split_index,split_val)
                #left call
                #right call
                return root
            else:
                #left call
                #right call


#    def test_step

