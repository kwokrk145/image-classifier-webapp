# Flask -> create the web app
# request -> access incoming data
# jsonify -> send JSON responses back to client
# render_template -> load and display HTML files 
# PIL -> Python Imaging Librray 
# -> a tool for opening, manipulating, and saving different file formats
# io -> provides tools for handling streams of data in memory
# -> you could read image data directly or convert raw bytes into a 
# -> PIL Image without saving it as a file
from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io
import joblib

# Load the model from joblib
model = joblib.load("digits_model.pkl")

# Creates Flask object, __name__ tells Flask where my app is located
app = Flask(__name__) 

# app.route("/") means when the user visits the homepage url, run the
# function below
@app.route('/')
def home():
    # return "<h1>Welcome to the image classifier app!</h1>"
    return render_template("index.html")

# First, notice that @app.route() means whenver a web request comes in 
# for this specific URL path, run the function right below this line.

# ./predict will listen for requests made to the URL ending with /predict
# -> You can think of predict as a different room in a house, or one of
# -> the rooms or page on my site. URLs are named to describe their 
# -> purpose. Think of / as the lobby and /predict as the kitchen for example

# methods=['POST'] means this route will ONLY respond to POST requests:
# -> POST requests are used when the client wants to send data to the
# -> server, like uploading a file or submitting a form. 

# TLDR: So, this function will run only if JS sends a POST request to
# https://.../predict with some sort of data.
@app.route("/predict", methods=["POST"])
def predict():
    # return jsonify({"prediction": "cat"})
    if "image" not in request.files:
        return "No image uploaded!", 400 # HTTP Status Code if fails
    
    file = request.files["image"]

    # Convert image to 8x8 grayscale and flatten
    # Image.open(file) opens the file, .convert("L") converts it to
    # grayscale mode, L stands for luminance. Then, .resize((8,8))
    # resizes the image to 8x8 pixels
    img = Image.open(file).convert("L").resize((8,8))

    # coverts the file into a NumPy array for easier handling
    img_arr = np.array(img)
    
    # Our training dataset ranges from 0-16 but the numpy pixels range
    # from 0-255, so we need to scale it down, which is why we divide.
    # But in our model 0 is white and 16 is black but our image is
    # designed to be the opposite way, which is why we subtract. 
    img_arr = 16 - (img_arr / 16)

    # .flatten() turns the 8x8 array into a 1D array with 64 values
    # reshape(1, -1) means make it 1 row and then the -1 means however 
    # many columns needed to satisfy the 1 row, so basically automatic
    # This is one sample point with automatic features (explanation below)
    img_flattened = img_arr.flatten().reshape(1, -1)

    # Now, that we have the array, we can predict
    # In machine learning, most models expect the input to be a 2D array
    # with this shape:
    # (number_of_samples, number_of_features). Each row is one sample, or
    # each list in this 2D array is one sample point and each column or feature
    # is how many columns in every sample point. 
    prediction = model.predict(img_flattened)[0]

    # Pass prediction to HTML template
    return render_template('results.html', prediction=prediction)

# More explanation on predict():
# -> request is an object Flask provides to access all incoming data
# -> from a client's request. request.files is a dictionary
# -> *Note that we use specifically image because the form is called
# -> image

if __name__ == '__main__':
    app.run(debug=True)