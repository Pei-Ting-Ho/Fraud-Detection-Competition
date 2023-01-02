# Data Preprocessing

## Basic Preprocessing
- File: **`preprocess_aggregation.py`**
- Features:
  - Primitive Features
    - Transform original features into some other features (Dimension Unchanged) 
    - Example: amount (A) * exchange rate = amount (B)
  - Aggregated Features
    - Group original features into some aggregation statistics (Dimension Reduced)
    - Example: customer's mean and standard deviation of purchase amounts
## Unsupervised Learning   
- File: **`preprocess_anomaly.py`**
- Features:
  - Categorical Encoding ([Target Encoder](https://contrib.scikit-learn.org/category_encoders/targetencoder.html))
  - Anomaly Detection ([IsolationForest Algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html))
