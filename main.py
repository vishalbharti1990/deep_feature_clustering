from PIL import Image
import numpy as np
import flask
from flask import render_template
import pickle
from keras.preprocessing.image import img_to_array
import tensorflow as tf
from sklearn.cluster import KMeans
from keras import Model
import io
from keras.applications.xception import preprocess_input as prpc_xce_inc
import keras.applications.xception as xce
from glob import glob

# initialize our Flask application and the KMeans model
app = flask.Flask(__name__)

# load the k-means and feature extraction models
def load_models():
	file = open('./model/clus_model_kmeans.pkl', 'rb')

	base_model = xce.Xception(weights='imagenet')
	xce_model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

	kmeans_model = pickle.load(file)

	graph = tf.get_default_graph()

	file.close()

	return kmeans_model, xce_model, graph

# get the kmeans model, xception model and tf graph
kmeans_model, xce_model, graph = load_models()

def prepare_image(image, target):

	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = prpc_xce_inc(image)

	# return the processed image
	return image

# This function takes as input the preprocessed input
# and returns the features extracted from xception network
def get_features(image):
	# extract features for current image
	with graph.as_default():
		features = np.ndarray.flatten(xce_model.predict(image))

	# return the extracted features
	return features.reshape(1,-1)


@app.route('/predict', methods=['POST'])
def get_closest_cluster_images():
	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("file"):
			# read the image in PIL format
			image = flask.request.files["file"].read()
			image = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification
			image = prepare_image(image, target=(299, 299))

			# get features from image
			features = get_features(image)

			cluster_id = kmeans_model.predict(features)

			# print(cluster_id)

			dir_name = './static/results/{}/'.format(cluster_id[0])

			closest_images_dir = glob(dir_name+'*.jpg')

			return_form = '<div class="row justify-content-md-center">\
						<div class="col-lg-6">\
						<h2>Top 5 closest fruits</h2>\
						<br>\
						<div class="result_div">\
						<img src="'+closest_images_dir[0]+'"class="rounded">\
						<img src="'+closest_images_dir[1]+'"class="rounded">\
						<img src="'+closest_images_dir[2]+'"class="rounded">\
						<img src="'+closest_images_dir[3]+'"class="rounded">\
						<img src="'+closest_images_dir[4]+'"class="rounded">\
						</div>\
						</div>\
						</div>'

			return return_form

@app.route("/")
@app.route("/index.html")
def render_home(appitem=None):
	return render_template("index.html")

if __name__ == '__main__':
	app.run(host='127.0.0.1', port=8080)