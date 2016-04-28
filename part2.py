import xml.etree.ElementTree as ET
from sklearn.cluster import KMeans
from scipy.spatial import distance

def createData(filename):
	tree = ET.parse(filename)
	root = tree.getroot()

	def parseAndAddArrayData(data):
		# Data is currently [x1, x2, x3, ... y1, y2, y3, ...]
		parsed_data = data.split()
		for j in range(len(parsed_data) / 2):
			n = j + len(parsed_data) / 2
			all_data[i].append((parsed_data[j], parsed_data[n]))

	# Set up data
	all_data = []
	i = 0
	for frame in root.findall('frame'):
		all_data.append([])

		for landmark in frame.findall('landmarks'):
			for feature in landmark.findall('data'):
				parseAndAddArrayData(feature.text)

		# print
		i += 1

	return all_data

def distance(p1, p2):
	return ((( float(p2[0]) - float(p1[0]) )**2) + (( float(p2[1]) - float(p1[1]) )**2))**0.5

def plotThatData():
	import numpy as np
	import matplotlib.pyplot as plt

	# Dataset to numpy array
	data = np.array(training_data[0])[:,0:2]

	# Plot
	N = len(data)
	labels = ['{0}'.format(i) for i in range(N)]
	plt.subplots_adjust(bottom = 0.1)
	plt.scatter(
	    data[:, 0], data[:, 1], marker = 'o',
	    cmap = plt.get_cmap('Spectral'))
	for label, x, y in zip(labels, data[:, 0], data[:, 1]):
	    plt.annotate(
	        label, 
	        xy = (x, y), xytext = (-20, 20),
	        textcoords = 'offset points', ha = 'right', va = 'bottom',
	        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
	        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

	plt.show()


# training_data = createData('../eye_move_lr.MP4_intraface_data.xml')
training_data = createData('../face_move_left_right[2].MP4_intraface_data.xml')

# These are the pairs from left to right features across the face
# All of these following numbers are 1 less than their documentation, because they are indices in the data array
pairs = [[4, 5], [3, 6], [2, 7], [1, 8], [0, 9],											# Eyebrows
		 [19, 28], [20, 27], [21, 26], [22, 25], [23, 30], [24, 29],						# Eyes
		 [14, 18], [24, 29],																# Nose
		 [31, 37], [32, 36], [33, 35], [42, 38], [41, 39], [41, 39], [43, 45], [46, 48]]	# Lips
middle = [10, 11, 12, 13, 16, 34, 40, 44, 47]

for pair in pairs:
	frame = 329
	mid_feature = 12
	dis1 = distance(training_data[frame][pair[0]], training_data[frame][mid_feature])
	dis2 = distance(training_data[frame][pair[1]], training_data[frame][mid_feature])
	print dis1, dis2, abs(dis1 - dis2)







# test_data = createData('../social.MP4_intraface_data.xml')
# # test_data = createData('../face_move_left_right.MP4_intraface_data.xml')

# estimator = KMeans(n_clusters=3)
# estimator.fit(training_data)
# labels = estimator.labels_
# print labels
# ans = estimator.predict(test_data)
# print ans
# for i in range(len(ans)):
# 	print ans[i],
### Forward = 2. Left (to camera) = 0. Right = 1

# for i in range(len(labels)):
# 	print labels[i], ":", training_data[i]

