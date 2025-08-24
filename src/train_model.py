import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def train_model():
    data_path = os.path.join('data', 'cleaned_flight_data.csv')
    df = pd.read_csv(data_path, parse_dates=['STD_datetime', 'ATD_datetime', 'STA_datetime', 'ATA_datetime'])

    df['departure_delay'] = (df['ATD_datetime'] - df['STD_datetime']).dt.total_seconds() / 60
    df.dropna(subset=['departure_delay'], inplace=True)

    df['hour'] = df['STD_datetime'].dt.hour
    df['day_of_week'] = df['STD_datetime'].dt.dayofweek
    df['Aircraft_Type'] = df['Aircraft'].apply(lambda x: x.split('(')[0].strip())

    features = ['hour', 'day_of_week', 'Aircraft_Type', 'To']
    target = 'departure_delay'

    X = df[features]
    y = df[target]

    categorical_features = ['Aircraft_Type', 'To']
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[('cat', one_hot_encoder, categorical_features)],
        remainder='passthrough'
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"Model R^2 score: {score:.2f}")

    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'delay_predictor.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_model()