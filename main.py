from Preprocess import preprocess_aggregation, preprocess_anomaly
from Model import ensemble_model


def main():
    
    print('STEP 1: PREPROCESSING')
    preprocess_aggregation.main()
    preprocess_anomaly.main()
    
    print('\n')
    print('STEP 2: MODELLING & PREDICTING')
    ensemble_model.main()


if __name__ == "__main__":
    main()