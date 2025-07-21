# torchvision.models -> gives access to pretrained models like ResNet
# torvision.transforms -> used to rsize, normalize, and prepare the 
# image for the model.
from flask import Flask, request, render_template
from PIL import Image
import torch
from torchvision import models, transforms

# Load the model from torchvision
# resent18 is a pretrained model 
# model.eval() puts the model in inference mode so it behaves correctly
# when predicting. Just make sure to run this when predicting since we
# aren't training right now.
model = models.resnet18(weights=True)
model.eval()

# Creates Flask object, __name__ tells Flask where my app is located
app = Flask(__name__) 

# Gets rid of newlines in the file, store values in classes list variable
with open("imagenet_classes.txt") as f:
    classes = [line.strip() for line in f]

# ResNet expect input images to be size 224x224, RGB, and normalized
# Resize(256) -> reduces the shorter edge of the image to 256 pixels, 
# while keeping the aspect ratio
# CenterCrop(224) -> crops the center 224 x 224 region
# ToTensor() -> A 224x224 RGB Image usually has shape height, width, then
# channel (number of colors so in this case 3 since RGB), so the image is
# stord as a 3D array: (224, 224, 3). Pytorch, however, expects (C, H, W),
# so all ToTensor() does is convert it to that format. 
# Normalize() -> adjusts pixel values to what ResNet expects
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# app.route("/") means when the user visits the homepage url, run the
# function below
@app.route('/')
def home():
    # return "<h1>Welcome to the image classifier app!</h1>"
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # return jsonify({"prediction": "cat"})
    if "image" not in request.files:
        return "No image uploaded!", 400 # HTTP Status Code if fails
    
    file = request.files["image"]

    
    img = Image.open(file).convert("RGB") # Ensure RGB format

    # transform(img) applies a series of image transformations, including
    # Resize(256), CenterCrop(224), and ToTensor()
    # The unsqueeze(0): this adds a batch dimension at position 0
    # -> a batch is just a group of images passed into a model at the same
    # -> time. Even when you have 1 image, the model still expects a batch
    # -> like shape. So you fake a batch of size 1 using .unsqueeze(0). 
    # -> Analogously, you can think of it as 1 batch is an order ticket, 
    # -> it can contain 1-20 meals and even if it's 1 item, it still needs
    # -> a tray (i.e. batch dimension), so the kitchen knows how to handle it
    img_tensor = transform(img).unsqueeze(0)

    # torch.no_grad() disables gradient tracking; normally PyTorch keeps track
    # of all operations but here we're just doing prediction so disabling it
    # saves memory. 
    # You then pass the image into the model, which returns a tensor of class scores
    # outputs is a tensor of shape [batch_size, num_classes]. Each row has 
    # scores for each class. So for example, if you had 3 classes and batch size 1:
    # outputs = [[-1.2, 2.8, 0.3]] wher class 0 has a score of -1.2, class 1 has a
    # a score of 2.8, etc. The 1 in .max() means for every image (which is 1 row), 
    # find the max across the columns. So every nested list is 1 image. .max(1) returns
    # values or the max scores and the indices of the max scores or the predictions.
    # We only care about the indices which is why we have _, predicted, meaning we 
    # ignore the first value. 
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = outputs.max(1)

    # This line converts the predicted clss index into a human readable label
    # Note that classes is the list of actual things that we got from our txt file
    prediction = classes[predicted.item()]
    
    # Pass prediction to HTML template
    return render_template('results.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)