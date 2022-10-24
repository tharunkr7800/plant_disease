from flask import Flask, render_template, request
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from PIL import Image
import base64
import io

app = Flask(__name__)

dict={0:'Apple___Apple_scab', 1:'Apple___Black_rot', 2:'Apple___Cedar_apple_rust', 3:'Apple___healthy', 4:'Blueberry___healthy', 5:'Cherry_(including_sour)___healthy', 6:'Cherry_(including_sour)___Powdery_mildew', 7:'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 8:'Corn_(maize)___Common_rust_', 9:'Corn_(maize)___healthy', 10:'Corn_(maize)___Northern_Leaf_Blight', 11:'Grape___Black_rot', 12:'Grape___Esca_(Black_Measles)', 13:'Grape___healthy', 14:'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 15:'Orange___Haunglongbing_(Citrus_greening)', 16:'Peach___Bacterial_spot', 17:'Peach___healthy', 18:'Pepper,_bell___Bacterial_spot', 19:'Pepper,_bell___healthy', 20:'Potato___Early_blight', 21:'Potato___healthy', 22:'Potato___Late_blight', 23:'Raspberry___healthy', 24:'Soybean___healthy', 25:'Squash___Powdery_mildew', 26:'Strawberry___healthy', 27:'Strawberry___Leaf_scorch', 28:'Tomato___Bacterial_spot', 29:'Tomato___Early_blight', 30:'Tomato___healthy', 31:'Tomato___Late_blight', 32:'Tomato___Leaf_Mold', 33:'Tomato___Septoria_leaf_spot', 34:'Tomato___Spider_mites Two-spotted_spider_mite', 35:'Tomato___Target_Spot', 36:'Tomato___Tomato_mosaic_virus', 37:'Tomato___Tomato_Yellow_Leaf_Curl_Virus'}

model = load_model('plant_disease.h5')

model.make_predict_function()

def predict_image(img_path):
    test_image=load_img(img_path, target_size=(150,150))
    test_image=img_to_array(test_image)
    test_image=test_image/255
    test_image = np.expand_dims(test_image, axis = 0)
    preds=np.argmax(model.predict(test_image))
    return dict[preds]

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "Sample_test/" + img.filename	
		img.save(img_path)

		p = predict_image(img_path)
		im = Image.open(img_path)
		data = io.BytesIO()
		im.save(data, "JPEG")
		encoded_img_data = base64.b64encode(data.getvalue())

	return render_template("index.html", prediction = p, img_path = encoded_img_data.decode('utf-8'))


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)