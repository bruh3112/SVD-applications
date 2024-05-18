import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import jupyter


#Line [1]
# Functions to represent make the vectorized image into a matrix or column vector
def packcw(A):
    return (A.T).reshape(-1,1)

def unpackcw(x,nr):
    return x.reshape(-1,nr).T

#Line [2]
# Load the YALE B database
YALE = io.loadmat('C:/Users/Hoang Anh/Downloads/YALEBXF.mat')

# divide into data and labels
X = YALE['X']
Y = YALE['Y']

# image dimensions
nr, nc = 192, 168  # height = 192, width = 168
dim = nr * nc  # dimension = 32256
nex = X.shape[1]  # Total number of examples = 2414

# data dimensions
nFc = 40  # Total number of faces
nImg = np.zeros(nFc)  # Numper of examples for each person
for i in range(nFc):  ##this counts the number of images subject 'i' has
    nImg[i] = (Y == i).sum()

# Show the faces
# Make board of 38 faces
faceIdx = 0  # index of face to display for each subject
Bh, Bw = 5, 8
FB = np.zeros((Bh * nr, Bw * nc))
for i in range(nFc):
    if nImg[i] > 0:
        loc = np.where(Y == i)[1]
        x = X[:, loc[faceIdx]]
        A = unpackcw(x, nr)
        row, col = divmod(i, Bw)
        rpt, cpt = row * nr, col * nc
        FB[rpt:rpt + nr, cpt:cpt + nc] = A

# plot images
plt.figure(figsize=(6, 6))
plt.imshow(FB, cmap='gray')
plt.axis('off')
plt.title("First Face Image of 38 Subjects (i=%i)" % faceIdx, fontsize=14)
plt.show()

print("The shape of the data is: ", X.shape)
print("")

# check number of images each subject has:
print("vector with number of images per subject: \n", nImg)# Load the YALE B database
YALE = io.loadmat('C:/Users/Hoang Anh/Downloads/YALEBXF.mat')

#divide into data and labels
X = YALE['X']
Y = YALE['Y']

#define image dimensions
nr, nc = 192, 168 # height = 192, width = 168
dim = nr * nc     # dimension = 32256
nex = X.shape[1]  # Total number of examples = 2414

#define data dimensions
nFc = 40             # Total number of faces
nImg = np.zeros(nFc) # Numper of examples for each person
for i in range(nFc): ##this counts the number of images subject 'i' has
    nImg[i] = (Y==i).sum()

# Show the faces
# Make board of 38 faces
faceIdx = 0 # index of face to display for each subject
Bh, Bw = 5, 8
FB = np.zeros((Bh*nr, Bw*nc))
for i in range(nFc):
    if nImg[i]>0:
        loc = np.where(Y==i)[1]
        x = X[:,loc[faceIdx]]
        A = unpackcw(x,nr)
        row, col = divmod(i,Bw)
        rpt, cpt = row*nr, col*nc
        FB[rpt:rpt+nr, cpt:cpt+nc] = A

#plot images
plt.figure(figsize = (6,6))
plt.imshow(FB, cmap='gray')
plt.axis('off')
plt.title("First Face Image of 38 Subjects (i=%i)" %faceIdx, fontsize=14)
plt.show()

print("The shape of the data is: ", X.shape)
print("")

#check number of images each subject has:
print("vector with number of images per subject: \n", nImg)


#Line [3]
# defining training (1900 samples) and testing (514 samples) data:

# training data and labels initializer
nTrainImages = 50 * nFc
trainData = np.zeros((32256, nTrainImages))
YTrain = np.zeros((1, nTrainImages))

# testing data and labels initializer
nTestImages = 514  # nTrainImages
testData = np.zeros((32256, nTestImages))
YTest = np.zeros((1, nTestImages))

# writting the new training and testing vectors
totalSamplesWritten = 0  # indicator for loop
print("shape test data:", testData.shape)

for i in range(nFc):
    if nImg[i] > 0:
        loc = np.where(Y == i)[1]
        trainData[:, (i * 50):(50 * (i + 1))] = X[:, loc[:50]]
        YTrain[:, (i * 50):(50 * (i + 1))] = Y[:, loc[:50]]

        nTrainSamples = loc.shape[0] - 50;  # counts number of samples in test set for each subject
        testData[:, (totalSamplesWritten):(totalSamplesWritten + nTrainSamples)] = X[:, loc[50:]]
        YTest[:, (totalSamplesWritten):(totalSamplesWritten + nTrainSamples)] = Y[:, loc[50:]]

        totalSamplesWritten += nTrainSamples

# removing empty samples (0th and 14th entry):
# removing empty faces (columns with only zeros)
zeroColumns = np.argwhere(np.all(trainData[..., :] == 0, axis=0))
trainData = np.delete(trainData, zeroColumns, axis=1)
YTrain = np.delete(YTrain, zeroColumns, axis=1)
print("shape train data:", trainData.shape)


#Line [4]
# getting the mean
meanTrain = trainData.mean()
print("mean of training data =", meanTrain)

#centering the data sets
cTrainData = trainData - meanTrain
cTestData = testData - meanTrain

#obtaining the SVD-decomposition
P, D, Q = np.linalg.svd(cTrainData, full_matrices=False)

#checking matrix decomposition was done correctly
test = P @ np.diag(D) @ Q
print("Data std:",np.std(cTrainData))
print("Reconstructed data std:",np.std(test))
print("Differences in std:",np.std(cTrainData - test))

import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=1000)

#defining vectors
var = D*D
cumSum = var.cumsum()

#plotting
plt.plot(var[:200])
plt.title('Top 200 Variances From Highest to Lowest')
plt.ylabel('Variance')
plt.xlabel('d')
plt.axvline(x=10, ymin=0, ymax=1,dash_capstyle='round')
plt.show()

#finding index where we capture 95% of the variance
index95 = (np.abs(cumSum - 0.95*cumSum[-1])).argmin()

plt.plot(cumSum[:200])
plt.title('Top 200 Variances Cumulative Sum')
plt.ylabel('Cumulative Variance')
plt.xlabel('d')
plt.axvline(x=index95, ymin=0, ymax=1,dash_capstyle='round')
plt.show()

def projectData(A,d):
    return np.diag(D)[:,:d] @ Q[:d,:]

data8 = projectData(cTrainData,8)
data16 = projectData(cTrainData,16)
data32 = projectData(cTrainData,32)
data64 = projectData(cTrainData,64)
data128 = projectData(cTrainData,128)
data256 = projectData(cTrainData,256)


##reconstructing the data from the projections

def reconstructData(projection):
    return P[:, :] @ projection + meanTrain


recon8 = reconstructData(data8)
recon16 = reconstructData(data16)
recon32 = reconstructData(data32)
recon64 = reconstructData(data64)
recon128 = reconstructData(data128)
recon256 = reconstructData(data256)


# Show the faces
def showFaces(data, d):
    # Make board of 38 faces
    faceIdx = 0  # index of face to display for each subject
    Bh, Bw = 5, 8  # size of face board
    FB = np.zeros((Bh * nr, Bw * nc))
    for i in range(nFc):
        if nImg[i] > 0:
            loc = np.where(YTrain == i)[1]
            x = data[:, loc[faceIdx]]
            A = unpackcw(x, nr)
            row, col = divmod(i, Bw)
            rpt, cpt = row * nr, col * nc
            FB[rpt:rpt + nr, cpt:cpt + nc] = A

    # plotting
    plt.figure(figsize=(6, 6))
    plt.imshow(FB, cmap='gray')
    plt.axis('off')
    plt.title("First Face Image of 38 Subjects (d=%d)" % d, fontsize=14)
    plt.show()


# displaing the faces
showFaces(recon8, 8)
showFaces(recon16, 16)
showFaces(recon32, 32)
showFaces(recon64, 64)
showFaces(recon128, 128)
showFaces(recon256, 256)


# Note nearest neighbor classifier is like K-nearest neighbors but with K=1

# identify the label of the nearest neighbor
def getLabel(trainData, testSample):
    nTrainSamples = trainData.shape[1]
    distanceLabelArray = np.zeros((nTrainSamples, 2))
    distanceLabelArray[:, 0] = YTrain  # adding labels to vector
    for i in range(nTrainSamples):
        distanceLabelArray[i, 1] = np.linalg.norm(testSample - trainData[:, i])  # adding distance to vector

    # sorting distances and the labels along with it
    distanceLabelArray = distanceLabelArray[distanceLabelArray[:, 1].argsort()]
    bestLabel = distanceLabelArray[0, 0]
    return bestLabel


# coding the classifier
def NNClassifer(trainData, testData):
    nTestSamples = testData.shape[1]
    resultLabels = np.zeros((1, nTestSamples))
    for j in range(nTestSamples):
        resultLabels[:, j] = getLabel(trainData, testData[:, j])  # get label of nearest neighbor
        # print("Percent of Data Classifed So Far:",j/nTestSamples)

    accuracy = np.sum(YTest == resultLabels) / nTestSamples
    return resultLabels, accuracy

labels, accuracy = NNClassifer(trainData, testData)

print("The labeling accuracy is", accuracy)

#calculating accuracy based on the nearest neighbor classifier and number of principal components
#WARNING: THIS TAKES A LONG TIME. Results are already loaded into the accuracyPCA.csv file.

labels8, accuracy8 = NNClassifer(recon8, testData)
labels16, accuracy16 = NNClassifer(recon16, testData)
labels32, accuracy32 = NNClassifer(recon32, testData)
labels64, accuracy64 = NNClassifer(recon64, testData)
labels128, accuracy128 = NNClassifer(recon128, testData)
labels256, accuracy256 = NNClassifer(recon256, testData)

acc = [accuracy8, accuracy16, accuracy32, accuracy64, accuracy128, accuracy256, accuracy]
np.savetxt("accuracyPCA.csv", acc, delimiter=",")

#loading accuracy results
acc = np.loadtxt(open("accuracyPCA.csv", "rb"), delimiter=",", skiprows=0)
accIndex = [8,16,32,64,128,256,1900]

#plotting results
plt.plot(accIndex[:6],acc[:6])
plt.title('Accuracy Vs Number of Principal Components')
plt.ylabel('Accuracy')
plt.xlabel('d')
plt.show()
