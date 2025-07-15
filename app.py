# Flask -> create the web app
# request -> access incoming data
# jsonify -> send JSON responses back to client
# render_template -> load and display HTML files 
from flask import Flask, request, jsonify, render_template

# Creates Flask object, __name__ tells Flask where my app is located
app = Flask(__name__) 

# app.route("/") means when the user visits the homepage url, run the
# function below
@app.route('/')
def home():
    return "<h1>Welcome to the image classifier app!</h1>"

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
    return jsonify({"prediction": "cat"})


if __name__ == '__main__':
    app.run(debug=True)