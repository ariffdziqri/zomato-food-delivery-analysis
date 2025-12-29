📦 Stacking Model for Delivery Time Prediction
This project builds a stacking regression model to predict delivery time (in minutes) using a combination of tree-based models and k-nearest neighbors, wrapped in a fully reproducible scikit-learn pipeline.
The model includes:
Data preprocessing (scaling + one-hot encoding)
A stacking ensemble of multiple regressors
Save / load functionality for reuse without retraining
📁 Project Structure
PROJECT/
│
├── code/
│   ├── cleaning.py      # Data cleaning and preprocessing
│   ├── model.py         # Stacking model definition
│
├── stacking.joblib      # Saved trained model (after training)
├── README.md
⚙️ Model Overview
The final model is a StackingRegressor with:
Base models
Gradient Boosting Regressor
Decision Tree Regressor
K-Nearest Neighbors Regressor
Final estimator
Random Forest Regressor
All preprocessing and modeling steps are combined into a single Pipeline.
📊 Expected Input
Target variable
Time_taken (min)
Feature inputs
You must specify:
features – all feature columns used by the model
numeric_f – numerical features (scaled)
cat_f – categorical features (one-hot encoded)
Example:
features = ["Distance", "Weather", "Traffic", "Vehicle_type"]
numeric_f = ["Distance"]
cat_f = ["Weather", "Traffic", "Vehicle_type"]
🚀 How to Train the Model
from model import BuildModel

bm = BuildModel(
    df=df,
    features=features,
    numeric_f=numeric_f,
    cat_f=cat_f
)

bm.fit()
print(bm.score())
This:
splits the data
trains the stacking model
evaluates performance on the test set
🔮 Making Predictions
preds = bm.predict(new_df[features])
The input DataFrame must contain the same feature columns used during training.
💾 Saving the Trained Model
bm.save("stacking.joblib")
This saves:
the trained pipeline
preprocessing steps
model parameters
feature configuration
No retraining is needed after saving.
📂 Loading a Saved Model
bm2 = BuildModel.load("stacking.joblib")
preds = bm2.predict(new_df[features])
This:
restores the exact trained model
allows predictions on new data
ensures full reproducibility
🔁 Reproducibility Notes
All models use a fixed random_state
The full preprocessing + model pipeline is saved
Predictions from a loaded model will match the original model exactly
To ensure reproducibility across machines, use the same library versions.
📦 Requirements
python >= 3.10
pandas
scikit-learn
joblib
(Optional) Pin versions in requirements.txt for strict reproducibility.
🧠 Notes
The model is designed for tabular regression problems
Categorical features are safely handled using handle_unknown="ignore"
The pipeline prevents data leakage by fitting preprocessing only on training data
