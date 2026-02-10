
import numpy as np
def entropy(y):
    class_labels , counts = np.unique(y,return_counts=True)
    probabilities = counts/len(y)
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy_value
 
class Node:
    def __init__(self,feature = None,threshold = None,left = None,right = None,*,value =None):
        self.feature = feature
        self.threshold = threshold
        self.left = left 
        self.right = right 
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None
    
    
    
    def entropy(y):
        class_labels , counts = np.unique(y,return_counts=True)
        probabilities = counts/len(y)
        entropy_value = -np.sum(probabilities * np.log2(probabilities))
        
        return entropy_value
    
class DecisionTree:
    def __init__(self,max_depth = 100):
        self.max_depth = max_depth
        self.root = None
        
    def info_gain(self,y,X_column,split_thresh):
        parent_entropy = entropy(y)
        left_indx = np.argwhere(X_column<=split_thresh).flatten()
        right_index = np.argwhere(X_column>= split_thresh).flatten()
        
        if len(left_indx) == 0 or len(right_index) == 0:
            return 0
        
        n = len(y)
        n_l,n_r = len(left_indx),len(right_index)
        e_l, e_r = entropy(y[left_indx]), entropy(y[right_index])
        
        child_entropy = (n_l/n)*e_l+(n_r/n)*e_r
        
        ig = parent_entropy - child_entropy
        return ig
    
    
    def best_split(self,X,y):
        best_gain = -1
        split_indx,split_thresh= None,None
        n_samples,n_features = X.shape
        
        for feat_indx in range(n_features):
            X_column = X[:,feat_indx]
            threshold = np.unique(X_column)
        for thr in threshold:
            gain = self.info_gain(y, X_column,thr)
            if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_indx
                    split_thresh = thr
        return split_idx,split_thresh
            
            
    def _build_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        
        if (depth >= self.max_depth) or (n_labels == 1) or (n_samples == 0):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        
        best_feat, best_thresh = self.best_split(X, y)

        
        left_idxs = np.argwhere(X[:, best_feat] <= best_thresh).flatten()
        right_idxs = np.argwhere(X[:, best_feat] > best_thresh).flatten()
        
        
        left_child = self._build_tree(X[left_idxs, :], y[left_idxs], depth+1)
        
        
        right_child = self._build_tree(X[right_idxs, :], y[right_idxs], depth+1)

        
        return Node(best_feat, best_thresh, left_child, right_child)

    def _most_common_label(self, y):
        
        from collections import Counter
        most_common = Counter(y).most_common(1)
        return most_common[0][0]

    def fit(self, X, y):
        
        self.root = self._build_tree(X, y)
    def predict(self, X):
        
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)

    def _traverse_tree(self, x, node):
        
        if node.is_leaf_node():
            return node.value

        
        if x[node.feature] <= node.threshold:
            
            return self._traverse_tree(x, node.left)
        else:
            
            return self._traverse_tree(x, node.right)
            





