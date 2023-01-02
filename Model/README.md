# Model Construction

## Feature Selection
- Method: Recursive Feature Elimination ([RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#))
- Motivation: Diverse Model Feature Sets

## Ensemble Models
- Model Calibration: Calibrated Classifier ([CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.htm))
- Boosting Framework: XGBoost, CatBoost, LightGBM
  ```
  param_1 = {
      'learning_rate': 0.05, 
      'n_estimators': 500, 
      'grow_policy': 'lossguide', 
      'max_depth': 8,
      'min_child_weight': 15,
      'subsample': 0.85,
      'colsample_bytree': 0.8, 
      'reg_alpha': 10, 
      'reg_lambda': 10, 
      'objective': 'binary:logistic',
      'scale_pos_weight': 10 
  } 
  XGBClassifier(**param_1)

  param_2 = {
      'learning_rate': 0.01, 
      'n_estimators': 500, 
      'grow_policy': 'Lossguide', 
      'max_depth': 8,
      'subsample': 0.85,
      'loss_function': 'Logloss', 
      'scale_pos_weight': 10
  } 
  CatBoostClassifier(**param_2)

  param_3 = {
      'boosting_type': 'goss', 
      'learning_rate': 0.05, 
      'n_estimators': 500, 
      'max_depth': 8,
      'subsample': 1.0,
      'colsample_bytree': 0.8,
      'reg_alpha': 3, 
      'reg_lambda': 3, 
      'objective': 'binary', 
      'scale_pos_weight': 10
  } 
  LGBMClassifier(**param_3)
  ```
