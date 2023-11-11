import numpy as np


def binSplitDataSet(dataSet, feature, value):
    """
    Binary Split DataSet
    :param dataSet:
    :param feature:
    :param value:
    :return:
    """
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    # if error: index 0 is out of bounds
    # mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :][0]
    # mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :][0]
    return mat0, mat1


# Tree node
def regLeaf(dataSet):
    """
    Generate leaf nodes
    Mean values of the target variable features in the regression tree
    :param dataSet:
    :return:
    """
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    """
    Error calculation function:
    Calculate squared error = mean squared error * total number of samples
    :param dataSet:
    :return:
    """
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(0, 1)):
    """
    Choose Best Split
    :param dataSet:
    :param leafType:
    :param errType:
    :param ops:
    :return:
    """
    # Parameter threshold for segmentation features
    tolS = ops[0]  # Error reduction value
    tolN = ops[1]  # Minimum number of samples to split
    # If all feature value are same, stop segmenting
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        # Stop Cutting v1: Remaining only one feature
        return None, leafType(dataSet)
        # If a good segmentation feature cannot be found
        # Directly use regLeaf to generate leaf nodes.
    m, n = np.shape(dataSet)
    S = errType(dataSet)  # Best features by calculating the average error
    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    # Iterate through each characteristic
    for featIndex in range(n - 1):
        # Iterate through each value
        for splitVal in set((dataSet[:, featIndex].T.A.tolist())[0]):
            # Binary Split Feature
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (
                np.shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:  # Update feature with the smallest error
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # Stop Cutting v2: If the error does not decrease significantly
    # Cancel the segmentation and create leaf nodes directly.
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # Stop Cutting v3: Calculate the size of the subset
    # If it is less than the minimum allowed number of samples stop cutting
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    # Return the feature number and feature value used for cutting
    return bestIndex, bestValue


# 构建tree
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(0, 1)):
    """
    Create Trees
    :param dataSet:
    :param leafType: Node type-regression tree
    :param errType: error calculation function
    :param ops: contains additional tuples required for tree construction
    :return:
    """
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    print(feat)
    if feat == None:
        return val  # Returns leaf node when met stopping condition
    # Assign value after cutting
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    # left and right subtrees after cutting
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


if __name__ == "__main__":
    myDat = [[1., 4.5], [2., 4.75], [3., 4.91], [4., 5.34], [5., 5.8],
             [6., 7.05], [7., 7.9], [8., 8.23], [9., 8.7], [10., 9.]]
    myDat = np.mat(myDat)
    print(createTree(myDat))

    # data plot
    import matplotlib.pyplot as plt
    plt.plot(myDat[:, 0], myDat[:, 1], 'ro')
    plt.show()
