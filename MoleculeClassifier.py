import re
import numpy as np
from imblearn.over_sampling import SMOTE
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
import time

# ***************************************************************
# ***************************************************************

trainLabels = []
trainVectors = []

testLabels = []
testVectors = []

predictionLabels = []

# creating a pattern for regex
pattern = re.compile('[0-9]+')


# ********************************************************************************************
# Reads the whole training data file and then:
#   - Stores data into a matrix
#   - Transforms regular matrix into a CSR sparse matrix
#   - Truncates sparse matrix to reduced number of features
#   - Returns the Truncated SVD matrix
# ********************************************************************************************
def getTrainData():
    labels = []
    vectors = []

    dataFile = open(r"train_drugs.dat", "r")

    curLine = dataFile.readline()  # reads first review
    while curLine:  # starts reading all reviews (one by one)

        vector = [0 for x in range(100001)] # creating the vector for one line (record) of the file (init. with 0's)
        dataRow = pattern.findall(curLine)  # extract all words (disposing of spaces/symbols)

        labels.append(int(dataRow[0])) # saves the label of the current record from training data
        del dataRow[0]                  # and deletes the element from the array to keep only features in matrix

        for record in dataRow:          #
            feature = int(record)       #  adding the 1's where they belong in the current row (record)
            vector[feature] = 1         #

        vectors.append(vector)          #  adding the complete row (with 0's and 1's) to the main matrix

        curLine = dataFile.readline()   # reading next line in the loop

    dataFile.close()

    vectorsNp = np.array(vectors)           #   in order to perform matrix operations we need to transform
    vectorsSparse = csr_matrix(vectorsNp)   #   our original matrix into a CSR sparse matrix

    myTsvd = TruncatedSVD(n_components=4000)    # Create a Truncated SVD object to reduce dimensions of sparse matrix

    vectorsTsvdFitted = myTsvd.fit(vectorsSparse, labels)   # Fits the LSI model of the training data into the tsvd
    vectorsTsvd = vectorsTsvdFitted.transform(vectorsSparse)    # now truncate matrix (reduce to desired # of dimensions)

    return vectorsTsvd, labels, vectorsTsvdFitted


# ********************************************************************************************
# Reads the whole test data file and then:
#   - Recieves the a sparse matrix (fitted to the LSI model from training data) as argument
#       and uses it to transform the test data matrix to the same model when truncating it
#       to the desired number of dimensions
#   - Stores data into a matrix
#   - Transforms regular matrix into a CSR sparse matrix
#   - Truncates sparse matrix to reduced number of features
#   - Returns the Truncated SVD matrix
# ********************************************************************************************
def getTestData(vectorsTsvdFitted):
    vectors = []

    dataFile = open(r"test.dat", "r")

    curLine = dataFile.readline()  # reads first review
    while curLine:  # starts reading all reviews (one by one)

        vector = [0 for x in range(100001)]    # the vector for one line (record) of the file
        dataRow = pattern.findall(curLine)  # extract all words (disposing of spaces/symbols)

        for record in dataRow:          #
            feature = int(record)       #  adding the 1's where they belong in the current row (record)
            vector[feature] = 1         #

        vectors.append(vector)          #  adding the complete row (with 0's and 1's) to the main matrix

        curLine = dataFile.readline()   # reading next line in the loop

    dataFile.close()

    vectorsNp = np.array(vectors)              #   in order to perform matrix operations we need to transform
    vectorsSparse = csr_matrix(vectorsNp)      #   our original matrix into a CSR sparse matrix

    vectorsTsvd = vectorsTsvdFitted.transform(vectorsSparse)

    return vectorsTsvd



def createTreeWithGini(vectorsTsvd, labels):
    # Creating the classifier object with desired characteristics
    classif = DecisionTreeClassifier(criterion="gini", max_depth=100, random_state=10, min_samples_leaf=4)

    # Fitiing the tree object to our training data
    classif.fit(vectorsTsvd, labels)

    return classif


def createBernoulli(vectorsTsvd, labels):
    # Creating the classifier object with desired characteristics
    classif = BernoulliNB()

    # Fitiing the tree object to our training data
    classif.fit(vectorsTsvd, labels)

    return classif


def createOutput(predictionLabels):
    predictionsDataFile = open(r"results","w")

    for prediction in predictionLabels:
        predictionsDataFile.write(str(int(prediction))+ '\n')

    predictionsDataFile.close()


def main():
    start = time.time()
    # Reading the training data from file and receiving a Transcated SVD that has been properly
    # fitted to an LSI model from the training data.
    trainVectorsTsvd, trainLabels, vectorsTsvdFitted = getTrainData()
    testVectorsTsvd = getTestData(vectorsTsvdFitted)

    # Handling the imbalanced data set with SMOTE
    smote = SMOTE(random_state=71)
    trainVectorsTsvd, trainLabels = smote.fit_sample(trainVectorsTsvd, trainLabels)

    # Creating the classifier and obtaining the predictions as an array of labels
    treeClassifier = createTreeWithGini(trainVectorsTsvd, trainLabels)
    preditionLabels = treeClassifier.predict(testVectorsTsvd)

#    naiveBayesClassifier = createBernoulli(trainVectorsTsvd, trainLabels)
#    preditionLabels = naiveBayesClassifier.predict(testVectorsTsvd)

    # Writing the training predictions into an output file called "results.dat"
    createOutput(preditionLabels)

    end = time.time()
    print(end - start)



if __name__ == "__main__":
    main()

#Decision Tree:
# max_dept=10, 1100 features, min_samples_split=2 = 71 score
# max_dept=10, 2000 features, min_samples_split=2 = 71 score
# max_dept=100, 2000 features, min_samples_split=2 = 76 score
# max_dept=500, 2000 features, min_samples_split=2 = 71 score
# max_dept=100, 2000 features, min_samples_split=20 = 72 score
# max_dept=100  4000 features, min_samples_split=2 =  69 score  (no smote)
# max_dept=100  4000 features, min_samples_split=2 =  81 score
# max_dept=100  4000 features, min_samples_leaf=20 =  80 score
# max_dept=100  4000 features, min_samples_leaf=60 =  72 score
# max_dept=100  4000 features, min_samples_leaf=8 =  77 score
# max_dept=100  4000 features, min_samples_leaf=4 =  75 score




#Bernoulli:
# features = 1000,   score = 60
# features = 2000,   score = 62
# features = 4000,   score = 65
# features = 10000,  Error = Unable to allocate 7.46 GiB for an array with shape (100001, 10010) and data type float64
