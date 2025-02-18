from flask import Flask, render_template, request
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(NeuralNet, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, 128)
    self.fc3 = nn.Linear(128, hidden_size)
    self.fc4 = nn.Linear(hidden_size, output_size)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    out = self.relu(out)
    out = self.fc3(out)
    out = self.relu(out)
    out = self.fc4(out)
    out = self.sigmoid(out)

    return out


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

model.eval()

feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from form
        try:
            input_data = [float(request.form[feature]) for feature in feature_names]
        except ValueError:
            return "Invalid input. Please enter valid numbers."

        # to NumPy and reshape
        input_array = np.array(input_data, dtype=np.float32).reshape(1, -1)

        scaler = StandardScaler()
        input_array = scaler.fit_transform(input_array)

        # NumPy to tensor
        input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(dim=1)

        # Predict using the model
        with torch.no_grad():
            prediction = model(input_tensor)

        #  to scalar
        prediction_value = prediction.item()
        result = "Malignant" if prediction_value > 0.5 else "Benign"

        return render_template("result.html", prediction=result)

    return render_template("index.html", features=feature_names)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input from form
        input_data = np.array([float(x) for x in request.form.values()], dtype=np.float32).reshape(1, -1)

        # scaling
        scaler = StandardScaler()
        input_data = scaler.fit_transform(input_data)  # Make sure to use the same scaler used in training

        # to tensor
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(dim=1)

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
        
        # output to scalar
        prediction = output.item()  
        result = "Malignant" if prediction > 0.5 else "Benign"

        return render_template("index.html", prediction=result)
    
    except Exception as e:
        return str(e)



if __name__ == "__main__":
    app.run(debug=True)
