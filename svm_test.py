""" 
Usage: 
face_recognize.py -i <test_image> 

Options: 
-h, --help					 Show this help 
-d, --train_dir =<train_dir> Directory with 
								images for training 
-i, --test_image =<test_image> Test image 
"""

# importing libraries 
import face_recognition
import docopt 
from sklearn import svm
import os 
import pickle
def face_recognize(test):
	filename = 'svm_trained_nov_5.sav'
	clf = pickle.load(open(filename, 'rb'))
	# Load the test image with unknown faces into a numpy array 
	test_image = face_recognition.load_image_file(test) 

	# Find all the faces in the test image using the default HOG-based model 
	face_locations = face_recognition.face_locations(test_image)
	no = len(face_locations)
	print("Number of faces detected: ", no)

	# Predict all the faces in the test image using the trained classifier 
	print("Found:")
	for i in range(no):
		test_image_enc = face_recognition.face_encodings(test_image)[i] 
		name = clf.predict([test_image_enc]) 
		print(*name)
        
def main(): 
	args = docopt.docopt(__doc__)
	test_image = args["--test_image"]
	face_recognize(test_image)
    
    
if __name__=="__main__": 
	main() 