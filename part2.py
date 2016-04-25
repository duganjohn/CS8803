import xml.etree.ElementTree as ET
from sklearn.cluster import KMeans

def createData(filename):
	tree = ET.parse(filename)
	root = tree.getroot()

	def parseAndAddArrayData(data):
		parsed_data = data.split()
		for j in range(len(parsed_data)):
			if j % 2 == 0:
				all_data[i].append((parsed_data[j], parsed_data[j + 1]))

	# Set up data
	all_data = []
	i = 0
	for frame in root.findall('frame'):
		all_data.append([])

		for landmark in frame.findall('landmarks'):
			for feature in landmark.findall('data'):
				# print feature.text.split()

				# print "feature", feature.text
				parseAndAddArrayData(feature.text)

		# print
		i += 1

	return all_data

training_data = createData('../eye_move_lr.MP4_intraface_data.xml')
print training_data[0]
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

