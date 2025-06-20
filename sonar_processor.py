
if __name__ == "__main__":
    filepath = "synthetic_sonar_dataset.csv"
    sample_rate = 1000  # Hz, must match the rate used in synthetic_sonar_data.py

    # 1. Load the dataset
    print("Loading synthetic sonar dataset...")
    X_raw, y = load_sonar_data(filepath)
    print("Dataset loaded successfully.")

    # 2. Preprocess signals and extract features
    print("Preprocessing signals and extracting features...")
    X_processed_features = []
    for index, row in X_raw.iterrows():
        signal = row.values
        preprocessed_signal = preprocess_signal(signal, sample_rate)
        features = extract_features(preprocessed_signal, sample_rate)
        X_processed_features.append(features)
    X_processed_features = pd.DataFrame(X_processed_features)
    print("Signal preprocessing and feature extraction complete.")
    print("Features head:\n", X_processed_features.head())

    # 3. Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed_features)
    X_scaled = pd.DataFrame(X_scaled, columns=X_processed_features.columns)
    print("Features scaled.")

    # 4. Split data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")

    # 5. Train and evaluate classification models
    train_and_evaluate_sonar_model(X_train, X_test, y_train, y_test, model_type='svm')
    train_and_evaluate_sonar_model(X_train, X_test, y_train, y_test, model_type='random_forest')
    train_and_evaluate_sonar_model(X_train, X_test, y_train, y_test, model_type='neural_network')


