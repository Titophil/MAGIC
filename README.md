🎯 MAGIC Gamma Telescope Classifier
This project demonstrates the application of K-Nearest Neighbors (KNN) and Naive Bayes (GaussianNB) classifiers to the MAGIC Gamma Telescope dataset. The goal is to classify events as gamma rays (signal) or hadron showers (background noise) using machine learning models.

📁 Dataset
Source: UCI Machine Learning Repository – MAGIC Gamma Telescope

Format: CSV

Features: 10 real-valued features extracted from telescope readings

Target:

g = gamma ray (positive class)

h = hadron (negative class)

📌 Project Overview
✔️ Workflow:
Data Loading
Read dataset and assign column names.

Preprocessing

Encode target (class) as binary (1 = gamma, 0 = hadron)

Visualize feature distributions

Split data into training (60%), validation (20%), and test (20%) sets

Apply Standard Scaling

Use Random Oversampling to balance training data

Modeling

Train a K-Nearest Neighbors classifier with n_neighbors=100

Train a Gaussian Naive Bayes classifier

Evaluate both models using classification reports

📊 Results
✅ KNN Classifier
Trained on oversampled & scaled data

Evaluated on test set

Uses n_neighbors = 100

✅ Naive Bayes Classifier
Fast and interpretable

Performs decently on standardized features

Both models are evaluated using metrics:

Precision

Recall

F1-score

Support

🧪 Dependencies
Make sure to install the following Python libraries:

bash
Copy
Edit
pip install pandas numpy matplotlib scikit-learn imbalanced-learn
🚀 Run the Code
Run in a Jupyter notebook or Google Colab environment:

python
Copy
Edit
# Load and preprocess data
df = pd.read_csv("magic04.data", names=cols)
...
Or directly open the Colab version here.

📬 Author
Titus Kiprono
LinkedIn | GitHub | Blog

