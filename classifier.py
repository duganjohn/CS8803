import xml.etree.ElementTree as ET
from sklearn.cluster import KMeans

def createData(filename):
	tree = ET.parse(filename)
	root = tree.getroot()

	def parseAndAddData(data):
		parsed_data = data.split()
		for datum in parsed_data:
			all_data[i].append(datum)

	# Set up data
	all_data = []
	i = 0
	for frame in root.findall('frame'):
		all_data.append([])

		for frameID in frame.findall('frameID'):
			# print "Frame:", frameID.text
			break

		for eye_status in frame.findall('eye_status'):
			# print "Eye Status", eye_status.text
			parseAndAddData(eye_status.text)
		
		for left_eye_vector in frame.findall('left_eye_gaze'):
			for l_vector in left_eye_vector.findall('data'):
				# print "Left eye vector", vector.text
				parseAndAddData(l_vector.text)
				break

		for right_eye_vector in frame.findall('right_eye_gaze'):
			for r_vector in right_eye_vector.findall('data'):
				# print "Right eye vector", vector.text
				parseAndAddData(r_vector.text)
				break
		# print
		i += 1

	return all_data


training_data = createData('../eye_move_lr.MP4_intraface_data.xml')
test_data = createData('../social.MP4_intraface_data.xml')
# test_data = createData('../face_move_left_right.MP4_intraface_data.xml')

estimator = KMeans(n_clusters=3)
estimator.fit(training_data)
labels = estimator.labels_
print labels
ans = estimator.predict(test_data)
print ans
for i in range(len(ans)):
	print ans[i],
### Forward = 2. Left (to camera) = 0. Right = 1

# for i in range(len(labels)):
# 	print labels[i], ":", training_data[i]

