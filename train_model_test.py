# load_digits is a built in dataset called digits
# -> a collection of small 8x8 pixel images of handwritten digits
# train_test_split lets you split the dataset into two parts
# -> on for training and one for testing its accuracy
# RandomForestClassifier is just for using Random Forest
# joblib is a library used to save and load trained ML models
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the full dataset
digits = load_digits()

# digits.data is a big NumPy array of shape (1797, 64) where each row
# is a digit image flattened into 64 pixel values. 
# digits.target is the correct digit labels for each row
X, y = digits.data, digits.target

# Essentially what we are doing here is teaching the model to recognize patterns.
# test_size = 0.2 means 20% of the data is used for testing,
# so 80% is used for training the model.

# X_train and y_train come from 80% of digits.data and digits.target.
# -> X_train contains the input patterns (pixel data from images).
# -> y_train contains the correct answers (the actual digits: 0–9).
# The model is trained on X_train and y_train, which means that it
# learns to map patterns (X_train) to correct digits (y_train).
# After training, we test the model using X_test and y_test.
# -> X_test contains new, unseen digit images.
# -> y_test contains the correct answers for those test images.
# We use X_test to see if the model can correctly predict digits it hasn't seen before,
# and we compare its predictions to y_test to measure how well it learned.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now we create the actual random forest object:
model = RandomForestClassifier()

# Next, we train the model
model.fit(X_train, y_train)

# Lastly, we save the model
joblib.dump(model, "digits_model.pkl")