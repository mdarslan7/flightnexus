import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import xgboost as xgb
import joblib
import os
import json
def create_enhanced_features(df):

    df['hour'] = df['STD_datetime'].dt.hour
    df['day_of_week'] = df['STD_datetime'].dt.dayofweek
    df['month'] = df['STD_datetime'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_peak_hour'] = df['hour'].isin([6, 7, 8, 9, 18, 19, 20, 21]).astype(int)

    df['scheduled_duration'] = (df['STA_datetime'] - df['STD_datetime']).dt.total_seconds() / 3600
    df['actual_duration'] = (df['ATA_datetime'] - df['ATD_datetime']).dt.total_seconds() / 3600

    df['departure_delay'] = (df['ATD_datetime'] - df['STD_datetime']).dt.total_seconds() / 60
    df['arrival_delay'] = (df['ATA_datetime'] - df['STA_datetime']).dt.total_seconds() / 60

    df = df[(df['departure_delay'] >= -60) & (df['departure_delay'] <= 360)]

    df = df.sort_values('STD_datetime').reset_index(drop=True)

    df['route'] = df['From'] + '_' + df['To']
    df['Aircraft_Type'] = df['Aircraft'].apply(lambda x: x.split('(')[0].strip() if pd.notna(x) else 'Unknown')

    df['route_avg_delay_7d'] = df.groupby('route')['departure_delay'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean().shift(1)
    ).fillna(0)
    df['aircraft_avg_delay_3d'] = df.groupby('Aircraft')['departure_delay'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
    ).fillna(0)

    df['flights_same_hour'] = df.groupby(['Date', 'hour']).cumcount() + 1
    df['total_flights_same_hour'] = df.groupby(['Date', 'hour'])['Flight Number'].transform('count')

    df_sorted = df.sort_values(['Aircraft', 'STD_datetime'])
    df_sorted['prev_arrival_delay'] = df_sorted.groupby('Aircraft')['arrival_delay'].shift(1).fillna(0)
    df_sorted['turnaround_time'] = df_sorted.groupby('Aircraft')['STD_datetime'].diff().dt.total_seconds() / 3600
    df_sorted['turnaround_time'] = df_sorted['turnaround_time'].fillna(24)

    domestic_routes = ['Delhi', 'Chennai', 'Bengaluru', 'Hyderabad', 'Kolkata', 'Ahmedabad', 'Pune', 'Goa']
    short_intl = ['Dubai', 'Doha', 'Abu Dhabi', 'Muscat', 'Colombo', 'Kathmandu', 'Dhaka']
    long_intl = ['London', 'Singapore', 'Bangkok', 'Istanbul', 'Frankfurt', 'New York']
    def get_route_complexity(destination):
        if any(city in destination for city in domestic_routes):
            return 1
        elif any(city in destination for city in short_intl):
            return 2
        elif any(city in destination for city in long_intl):
            return 3
        else:
            return 2
    df_sorted['route_complexity'] = df_sorted['To'].apply(get_route_complexity)
    print(f"Enhanced features created. Dataset shape: {df_sorted.shape}")
    print(f"Clean delay data points: {len(df_sorted)}")
    return df_sorted
def train_model():
    """
    Trains an enhanced ensemble model to predict flight departure delays.
    """
    print("--- Enhanced Model Training Started ---")
    data_path = os.path.join('data', 'cleaned_flight_data.csv')
    df = pd.read_csv(data_path, parse_dates=['STD_datetime', 'ATD_datetime', 'STA_datetime', 'ATA_datetime'])
    print(f"Original dataset size: {len(df)}")

    df = create_enhanced_features(df)

    df.dropna(subset=['departure_delay'], inplace=True)

    feature_columns = [
        'hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour',
        'scheduled_duration', 'flights_same_hour', 'total_flights_same_hour',
        'route_avg_delay_7d', 'aircraft_avg_delay_3d', 'prev_arrival_delay',
        'turnaround_time', 'route_complexity'
    ]

    le_aircraft = LabelEncoder()
    le_route = LabelEncoder()
    le_aircraft_type = LabelEncoder()
    df['aircraft_encoded'] = le_aircraft.fit_transform(df['Aircraft'].astype(str))
    df['route_encoded'] = le_route.fit_transform(df['route'].astype(str))
    df['aircraft_type_encoded'] = le_aircraft_type.fit_transform(df['Aircraft_Type'].astype(str))
    feature_columns.extend(['aircraft_encoded', 'route_encoded', 'aircraft_type_encoded'])

    X = df[feature_columns].fillna(0)
    y = df['departure_delay']
    print(f"Training with {len(X)} samples and {len(feature_columns)} features")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'rf': RandomForestRegressor(random_state=42),
        'gb': GradientBoostingRegressor(random_state=42),
        'xgb': xgb.XGBRegressor(random_state=42, eval_metric='rmse')
    }

    param_grids = {
        'rf': {
            'n_estimators': [100, 200],
            'max_depth': [10, 15],
            'min_samples_split': [2, 5]
        },
        'gb': {
            'n_estimators': [100, 150],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.2]
        },
        'xgb': {
            'n_estimators': [100, 150],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.2]
        }
    }

    best_models = {}
    print("\n--- Training Individual Models ---")
    for name, model in models.items():
        print(f"Training {name.upper()}...")
        grid_search = GridSearchCV(
            model, param_grids[name], 
            cv=3, scoring='neg_mean_absolute_error', 
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        score = -grid_search.best_score_
        print(f"Best {name.upper()} MAE: {score:.2f} minutes")

    print("\n--- Creating Ensemble Model ---")
    ensemble = VotingRegressor([
        ('rf', best_models['rf']),
        ('gb', best_models['gb']),
        ('xgb', best_models['xgb'])
    ])
    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"\n--- Ensemble Model Performance ---")
    print(f"MAE: {mae:.2f} minutes")
    print(f"RMSE: {rmse:.2f} minutes")
    print(f"R²: {r2:.3f}")

    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_models['rf'].feature_importances_
    }).sort_values('importance', ascending=False)
    print(f"\n--- Top 10 Most Important Features ---")
    for _, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']}: {row['importance']:.3f}")

    os.makedirs('models', exist_ok=True)

    ensemble_path = os.path.join('models', 'ensemble_delay_predictor.joblib')
    joblib.dump(ensemble, ensemble_path)

    joblib.dump(le_aircraft, os.path.join('models', 'aircraft_encoder.joblib'))
    joblib.dump(le_route, os.path.join('models', 'route_encoder.joblib'))
    joblib.dump(le_aircraft_type, os.path.join('models', 'aircraft_type_encoder.joblib'))
    joblib.dump(feature_columns, os.path.join('models', 'feature_columns.joblib'))

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'features_count': len(feature_columns)
    }
    with open(os.path.join('models', 'model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    feature_importance.to_csv(os.path.join('models', 'feature_importance.csv'), index=False)
    print(f"\n--- Models and Metrics Saved ---")
    print(f"Ensemble model saved to {ensemble_path}")
    print(f"Model metrics saved to models/model_metrics.json")

    legacy_model_path = os.path.join('models', 'delay_predictor.joblib')
    joblib.dump(best_models['gb'], legacy_model_path)
    print(f"Legacy model saved to {legacy_model_path}")
    return ensemble
if __name__ == '__main__':
    train_model()