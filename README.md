# Fraud-Detection-Competition

## Problem Statement
### Overview
Given the customer information and transactional records, we are tasked with predicting whether at a specific date, we should file Suspicious Activity Reports (SARs) with reagrds to potential instances of money laundering.
This is a problem with great importance, as nowadays it has become harder for humans to sift through large amounts of financial transactions and pick out the suspicious cases. 
Hence, we could resort to the partially automated approach, based on defined assessment criteria and / or unsupervised anomaly detection, to help us filter out the most probable positive cases for detailed review.  

### Datasets
- Transactional records of various kinds 
- Customer information
- Alert keys and dates

### Evaluation
- Numerator = _n - 1_ (Total positives - 1)
- Denominator = _i_ (Total number of cases, when probabilities are sorted in the descending order, needed to detect _n - 1_ positives) 

## Suggested Approach
### Step 1: Data Preprocessing
#### Step 1.1: Basic Preprocessing
- Data Characteristics
  - Transactional records (One-To-Many Relationship): Each customer might have >=0 row of a specific transactional kind.
  - Customer information (One-To-One Relationship): Each customer has =1 row of associated attributes.
  - Alert keys (One-To-Many Relationship): Each customer might have >=1 row of associated incidents.
- Feature Engineering 
  - Primitive features
  - Aggregated features
#### Step 1.2: Unsupervised Learning   

### Step 2: Model Construction
#### Step 2.1: Feature Selection
#### Step 2.2: Ensemble Models

### Step 3: Final Compilation
- **`./Preprocess`**: Preprocessing scripts
- **`./Model`**: Modelling script
- **`main.py`**: Execution script
