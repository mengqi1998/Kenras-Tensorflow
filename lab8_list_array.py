from numpy import array

features = [[0, 0], [1, 1], [2, 2]]
labels = [1, 4, 8]
featuresArray = array(features)
print(type(featuresArray))
print(featuresArray)
print(featuresArray[0])
print(featuresArray[1])
print(featuresArray[2])
print(featuresArray[:,0])
print(featuresArray[:,1])

