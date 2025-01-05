import numpy as np 

class Node:
    def __init__(self, feature=None,featureValue=None,leftPointer=None,rightPointer=None, classLabel=None):
        '''
        Building a Node of the decision tree 
        '''
        self.feature = feature # denotes the index of column in data matrix
        self.featureValue = featureValue # the value of the feature (to compare)
        self.leftPointer = leftPointer
        self.rightPointer = rightPointer
        
        # if the node is leaf node, it also contains the classLabel
        self.classLabel = classLabel


class Tree():
    
    def __init__(self,dataset,labels):
        self.data = dataset
        self.labels = labels
        self.root = None
    
    def CalcEntropy(self, data):
        counts = np.bincount(data)
        probability = counts/len(data)
        
        entropy = -np.sum([p * np.log2(p) for p in probability if p > 0])
        return entropy
    
    def split_data(self, data , labels , feature , value):
        rightIndices = np.where(data[: , feature] > value)[0]
        leftIndices = np.where(data[:  , feature] <= value)[0]
        
        dataRight = data[rightIndices]
        dataLeft = data[leftIndices]
        labelRight = labels[rightIndices]
        labelLeft = labels[leftIndices]
        
        return dataRight, labelRight , dataLeft , labelLeft
    
    def buildTree(self, data=None, labels=None):
        data = data if data is not None else self.data
        labels = labels if labels is not None else self.labels

        # base case : Leaf Node 
        if len(set(labels)) == 1:
            return Node(classLabel=labels[0])
        
        DataSetEntropy = self.CalcEntropy(labels)
        
        bestfeature=None    # the index of the best feature 
        bestValue=None
        bestgain=-np.inf 
        bestSplit=None
        
        n_features = data.shape[1]
        
        for feature in range(n_features):
            unique_feature_values = np.unique(data[: , feature])
            for value in unique_feature_values:
                #split at this feature and value 
                dataRight, labelRight , dataLeft , labelLeft = self.split_data(data,labels,feature, value)
                
                # Skip invalid splits
                if len(labelRight) == 0 or len(labelLeft) == 0:
                    continue
                    
                RightEntropy = self.CalcEntropy(labelRight)
                LeftEntropy = self.CalcEntropy(labelLeft)
                
                gain = DataSetEntropy - (
                    (len(labelRight)/len(labels)) * RightEntropy
                    + 
                    (len(labelLeft)/len(labels)) * LeftEntropy
                    )
                
                if gain > bestgain:
                    bestgain = gain 
                    bestfeature = feature
                    bestValue = value
                    bestSplit = (dataRight, labelRight , dataLeft , labelLeft)
                    
        # If no valid split was found, return a leaf node with the majority label
                
        if bestSplit == None : 
            majority_label = np.bincount(labels).argmax()
            return Node(classLabel=majority_label)
            
        # Build the tree recursively 
        
        dataRight, labelRight , dataLeft , labelLeft = bestSplit
        rightChild = self.buildTree(dataRight,labelRight)
        leftChild = self.buildTree(dataLeft , labelLeft)
        
        node = Node(feature=bestfeature,featureValue=bestValue,leftPointer=leftChild,rightPointer=rightChild)
        
        if data is self.data:
            self.root = node 
        
        return node
    
    def predict(self, tree, sample):
        if tree.classLabel is not None:
            return tree.classLabel
            
        if sample[tree.feature] <= tree.featureValue:
            return self.predict(tree.leftPointer, sample)
        return self.predict(tree.rightPointer, sample)



if __name__ == '__main__':
    X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    y = np.array([1, 1, 0, 0])
    
    decisionTree= Tree(X,y)
    decisionTree.buildTree()
    
    sample = np.array([1, 1])
    prediction = decisionTree.predict(decisionTree.root,sample)
    
    print(f"Prediction for sample {sample}: {prediction}")  
    
    