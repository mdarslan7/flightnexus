import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import google.generativeai as genai

st.set_page_config(
    page_title="Flight Scheduling Assistant",
    page_icon="✈️",
    layout="wide",
)

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except (KeyError, AttributeError):
    st.warning("GEMINI_API_KEY not found in secrets.toml. NLP features disabled.", icon="⚠️")
    genai = None

@st.cache_data
def load_data():
    flight_data_path = os.path.join('data', 'cleaned_flight_data.csv')
    critical_flights_path = os.path.join('data', 'critical_flights.csv')

    if not os.path.exists(flight_data_path) or not os.path.exists(critical_flights_path):
        st.error("Data files not found! Please run the backend pipeline first by executing 'python -m src.main' in your terminal.", icon="🚨")
        return None, None

    df = pd.read_csv(flight_data_path, parse_dates=['STD_datetime', 'ATD_datetime', 'STA_datetime', 'ATA_datetime'])
    df_critical = pd.read_csv(critical_flights_path)
    return df, df_critical

@st.cache_resource
def load_model():
    """Loads the trained delay prediction model (enhanced ensemble if available)."""
    # Try to load enhanced ensemble model first
    ensemble_path = os.path.join('models', 'ensemble_delay_predictor.joblib')
    legacy_path = os.path.join('models', 'delay_predictor.joblib')
    
    if os.path.exists(ensemble_path):
        try:
            # Load ensemble model and related encoders
            model = joblib.load(ensemble_path)
            
            # Load encoders for enhanced features
            encoders = {}
            encoder_files = ['aircraft_encoder.joblib', 'route_encoder.joblib', 'aircraft_type_encoder.joblib']
            for encoder_file in encoder_files:
                encoder_path = os.path.join('models', encoder_file)
                if os.path.exists(encoder_path):
                    encoder_name = encoder_file.replace('.joblib', '')
                    encoders[encoder_name] = joblib.load(encoder_path)
            
            # Load feature columns
            feature_columns_path = os.path.join('models', 'feature_columns.joblib')
            if os.path.exists(feature_columns_path):
                feature_columns = joblib.load(feature_columns_path)
            else:
                feature_columns = None
                
            return {'model': model, 'encoders': encoders, 'feature_columns': feature_columns, 'type': 'ensemble'}
            
        except Exception as e:
            st.warning(f"Failed to load ensemble model: {e}. Falling back to legacy model.", icon="⚠️")
    
    # Fall back to legacy model
    if os.path.exists(legacy_path):
        model = joblib.load(legacy_path)
        return {'model': model, 'type': 'legacy'}
    else:
        st.error("No model files found! Please run the backend pipeline first by executing 'python -m src.main' in your terminal.", icon="🚨")
        return None

# Load all assets
# Load all assets
df, df_critical = load_data()
model_data = load_model()

# --- HELPER FUNCTION FOR NLP ---
def get_gemini_response(question, df):
    """Generates a response from Gemini API."""
    if not genai:
        return "Gemini API is not configured. Please add your key to the .streamlit/secrets.toml file."
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Create comprehensive data summary instead of just head()
    data_summary = f"""
    Dataset Overview:
    - Total flights: {len(df)}
    - Date range: {df['Date'].min()} to {df['Date'].max()}
    - Unique destinations: {df['To'].nunique()} ({', '.join(sorted(df['To'].unique()[:10]))}{'...' if df['To'].nunique() > 10 else ''})
    - Unique aircraft: {df['Aircraft'].nunique()}
    - Airlines: {df['Flight Number'].str.extract(r'^([A-Z]+)').iloc[:, 0].nunique() if not df['Flight Number'].isna().all() else 'N/A'}
    
    Sample data (first 5 rows):
    {df.head().to_markdown()}
    
    Key Statistics:
    - Average departure delay: {((df['ATD_datetime'] - df['STD_datetime']).dt.total_seconds() / 60).mean():.2f} minutes
    - Average arrival delay: {((df['ATA_datetime'] - df['STA_datetime']).dt.total_seconds() / 60).mean():.2f} minutes
    - Busiest hour: {df['STD_datetime'].dt.hour.mode().iloc[0]}:00
    """
    
    prompt = f"""You are a helpful flight data analyst for Mumbai Airport (BOM). Your task is to answer the user's question based on the provided comprehensive flight data summary.

    Data columns are: {', '.join(df.columns)}
    
    {data_summary}
    
    User Question: '{question}'
    
    Please provide accurate, data-driven insights based on the complete dataset information provided above."""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred with the Gemini API: {e}"

# --- PREDICTION HELPER FUNCTION ---
def make_prediction(hour, to_dest, aircraft, model_data, df):
    """Make prediction using either ensemble or legacy model"""
    
    if model_data['type'] == 'ensemble':
        # Enhanced prediction with realistic feature values
        try:
            # Calculate realistic feature values from actual data
            avg_departure_delay = df['departure_delay'].mean() if 'departure_delay' in df.columns else 0
            if pd.isna(avg_departure_delay):
                avg_departure_delay = 0
                
            # Calculate average values for more realistic predictions
            route_key = f"Mumbai (BOM)_{to_dest}"
            route_delays = df[df['To'] == to_dest]['departure_delay'] if 'departure_delay' in df.columns else [5]
            route_avg = route_delays.mean() if len(route_delays) > 0 and not pd.isna(route_delays.mean()) else 5
            
            aircraft_delays = df[df['Aircraft'] == aircraft]['departure_delay'] if 'departure_delay' in df.columns else [3]
            aircraft_avg = aircraft_delays.mean() if len(aircraft_delays) > 0 and not pd.isna(aircraft_delays.mean()) else 3
            
            # Count flights at this hour from data
            hour_flights = df[df['STD_datetime'].dt.hour == hour] if 'STD_datetime' in df.columns else []
            flights_this_hour = len(hour_flights) if len(hour_flights) > 0 else 3
            
            # Create realistic input dataframe
            input_data = {
                'hour': [hour],
                'day_of_week': [1],  # Tuesday (more realistic than Monday)
                'month': [7],  # July 
                'is_weekend': [0],
                'is_peak_hour': [1 if hour in [6, 7, 8, 9, 18, 19, 20, 21] else 0],
                'scheduled_duration': [1.5],  # More realistic 1.5 hours
                'flights_same_hour': [min(flights_this_hour, 8)],  # Realistic based on data
                'total_flights_same_hour': [min(flights_this_hour * 2, 15)],  # Reasonable total
                'route_avg_delay_7d': [max(0, min(route_avg, 30))],  # Cap at 30 minutes
                'aircraft_avg_delay_3d': [max(0, min(aircraft_avg, 20))],  # Cap at 20 minutes
                'prev_arrival_delay': [2],  # Low previous delay
                'turnaround_time': [4],  # 4 hours turnaround
                'route_complexity': [1 if 'India' in to_dest or any(city in to_dest for city in ['Delhi', 'Chennai', 'Bengaluru']) else 2],  # Domestic vs international
            }
            
            # Encode categorical variables with better error handling
            encoders = model_data['encoders']
            
            if 'aircraft_encoder' in encoders:
                try:
                    aircraft_encoded = encoders['aircraft_encoder'].transform([aircraft])[0]
                except:
                    # Use a middle value from the encoder's classes
                    try:
                        aircraft_encoded = len(encoders['aircraft_encoder'].classes_) // 2
                    except:
                        aircraft_encoded = 0
                input_data['aircraft_encoded'] = [aircraft_encoded]
            else:
                input_data['aircraft_encoded'] = [0]
            
            if 'route_encoder' in encoders:
                route = f"Mumbai (BOM)_{to_dest}"
                try:
                    route_encoded = encoders['route_encoder'].transform([route])[0]
                except:
                    # Use a middle value from the encoder's classes
                    try:
                        route_encoded = len(encoders['route_encoder'].classes_) // 2
                    except:
                        route_encoded = 0
                input_data['route_encoded'] = [route_encoded]
            else:
                input_data['route_encoded'] = [0]
            
            if 'aircraft_type_encoder' in encoders:
                aircraft_type = aircraft.split('(')[0].strip()
                try:
                    aircraft_type_encoded = encoders['aircraft_type_encoder'].transform([aircraft_type])[0]
                except:
                    # Use a middle value from the encoder's classes
                    try:
                        aircraft_type_encoded = len(encoders['aircraft_type_encoder'].classes_) // 2
                    except:
                        aircraft_type_encoded = 0
                input_data['aircraft_type_encoded'] = [aircraft_type_encoded]
            else:
                input_data['aircraft_type_encoded'] = [0]
            
            input_df = pd.DataFrame(input_data)
            
            # Make prediction
            prediction = model_data['model'].predict(input_df)
            
            # Apply reasonable bounds to the prediction
            bounded_prediction = max(0, min(prediction[0], 120))  # Cap between 0-120 minutes
            
            return bounded_prediction, "ensemble"
            
        except Exception as e:
            st.warning(f"Ensemble prediction failed: {e}. Using legacy model.")
            # Fall back to legacy prediction
            pass
    
    # Legacy model or fallback prediction
    try:
        input_df = pd.DataFrame({
            'hour': [hour],
            'day_of_week': [1],  # Use Tuesday instead of Monday
            'Aircraft_Type': [aircraft.split('(')[0].strip()],
            'To': [to_dest],
            'Aircraft': [aircraft]
        })
        
        if model_data['type'] == 'ensemble':
            # Use the gradient boosting component of ensemble as fallback
            prediction = model_data['model'].estimators_[1].predict(input_df)  # GB model
        else:
            prediction = model_data['model'].predict(input_df)
            
        return max(0, min(prediction[0], 90)), "legacy"  # Apply bounds
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return 15.0, "default"  # Reasonable default prediction

# --- UI LAYOUT ---
st.title("✈️ AI-Powered Flight Scheduling Assistant")
st.markdown("An interactive dashboard to analyze flight data, predict delays, and identify critical flights at Mumbai Airport (BOM).")

# Stop the app if data is not loaded
if df is None or df_critical is None or model_data is None:
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Airport Overview", "⚙️ Schedule Tuner", "🚨 Critical Flights", "💬 Ask Gemini", "🔬 Model Insights"])

with tab1:
    st.header("Airport Overview: Busiest Times and Delays")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Busiest Times at the Airport")
        df['hour'] = df['STD_datetime'].dt.hour
        hourly_activity = df.groupby('hour').size()
        fig1, ax1 = plt.subplots()
        hourly_activity.plot(kind='bar', ax=ax1, color='skyblue', title='Airport Activity by Hour')
        ax1.set_xlabel('Hour of the Day')
        ax1.set_ylabel('Number of Flights')
        ax1.tick_params(axis='x', rotation=0)
        st.pyplot(fig1)
    with col2:
        st.subheader("Average Delays by Hour")
        df['departure_delay'] = (df['ATD_datetime'] - df['STD_datetime']).dt.total_seconds() / 60
        df['arrival_delay'] = (df['ATA_datetime'] - df['STA_datetime']).dt.total_seconds() / 60
        hourly_delays = df.groupby('hour')[['departure_delay', 'arrival_delay']].mean()
        fig2, ax2 = plt.subplots()
        hourly_delays.plot(kind='line', ax=ax2, style='.-', marker='o', title='Average Flight Delays by Hour')
        ax2.set_xlabel('Hour of the Day')
        ax2.set_ylabel('Average Delay (minutes)')
        ax2.legend(['Departure Delay', 'Arrival Delay'])
        ax2.grid(True)
        st.pyplot(fig2)
    
    st.subheader("🎯 Best Times for Takeoff/Landing Analysis")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**Departure Performance by Hour**")

        df['on_time_departure'] = df['departure_delay'] <= 15  
        
        departure_performance = df.groupby('hour').agg({
            'departure_delay': 'mean',
            'on_time_departure': 'mean'
        }).round(2)
        
        departure_performance['on_time_percentage'] = (departure_performance['on_time_departure'] * 100).round(1)
        
        fig3, ax3 = plt.subplots(figsize=(10, 6))

        bars = ax3.bar(departure_performance.index, departure_performance['on_time_percentage'], 
                      color=['green' if x >= 80 else 'orange' if x >= 60 else 'red' 
                             for x in departure_performance['on_time_percentage']],
                      alpha=0.7)
        
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('On-Time Departure Percentage (%)')
        ax3.set_title('Best Departure Times (Scheduled vs Actual)')
        ax3.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Good Performance (80%+)')
        ax3.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='Fair Performance (60%+)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, departure_performance['on_time_percentage']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value}%', ha='center', va='bottom', fontsize=8)
        
        st.pyplot(fig3)
        
        best_departure_hours = departure_performance.nlargest(3, 'on_time_percentage').index.tolist()
        worst_departure_hours = departure_performance.nsmallest(3, 'on_time_percentage').index.tolist()
        
        st.success(f"🟢 **Best Departure Times**: {', '.join([f'{h:02d}:00' for h in best_departure_hours])}")
        st.error(f"🔴 **Avoid Departure Times**: {', '.join([f'{h:02d}:00' for h in worst_departure_hours])}")
    
    with col4:
        st.markdown("**Arrival Performance by Hour**")

        df['on_time_arrival'] = df['arrival_delay'] <= 15  
        df['arrival_hour'] = df['STA_datetime'].dt.hour
        
        arrival_performance = df.groupby('arrival_hour').agg({
            'arrival_delay': 'mean',
            'on_time_arrival': 'mean'
        }).round(2)
        
        arrival_performance['on_time_percentage'] = (arrival_performance['on_time_arrival'] * 100).round(1)

        fig4, ax4 = plt.subplots(figsize=(10, 6))
        
        bars2 = ax4.bar(arrival_performance.index, arrival_performance['on_time_percentage'], 
                       color=['green' if x >= 80 else 'orange' if x >= 60 else 'red' 
                              for x in arrival_performance['on_time_percentage']],
                       alpha=0.7)
        
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('On-Time Arrival Percentage (%)')
        ax4.set_title('Best Landing Times (Scheduled vs Actual)')
        ax4.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Good Performance (80%+)')
        ax4.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='Fair Performance (60%+)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        for bar, value in zip(bars2, arrival_performance['on_time_percentage']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value}%', ha='center', va='bottom', fontsize=8)
        
        st.pyplot(fig4)
        
        best_arrival_hours = arrival_performance.nlargest(3, 'on_time_percentage').index.tolist()
        worst_arrival_hours = arrival_performance.nsmallest(3, 'on_time_percentage').index.tolist()
        
        st.success(f"🟢 **Best Landing Times**: {', '.join([f'{h:02d}:00' for h in best_arrival_hours])}")
        st.error(f"🔴 **Avoid Landing Times**: {', '.join([f'{h:02d}:00' for h in worst_arrival_hours])}")
    
    st.markdown("---")
    st.subheader("📋 Scheduling Recommendations")
    
    col5, col6 = st.columns(2)
    with col5:
        avg_departure_delay = df['departure_delay'].mean()
        best_dep_hour = departure_performance['departure_delay'].idxmin()
        best_dep_delay = departure_performance.loc[best_dep_hour, 'departure_delay']
        
        st.metric("Overall Avg Departure Delay", f"{avg_departure_delay:.1f} min")
        st.metric("Best Departure Hour", f"{best_dep_hour:02d}:00", f"{best_dep_delay:.1f} min delay")
    
    with col6:
        avg_arrival_delay = df['arrival_delay'].mean()
        best_arr_hour = arrival_performance['arrival_delay'].idxmin()
        best_arr_delay = arrival_performance.loc[best_arr_hour, 'arrival_delay']
        
        st.metric("Overall Avg Arrival Delay", f"{avg_arrival_delay:.1f} min")
        st.metric("Best Arrival Hour", f"{best_arr_hour:02d}:00", f"{best_arr_delay:.1f} min delay")

with tab2:
    st.header("Schedule Tuning Model")
    st.markdown("Predict the potential departure delay for a new or rescheduled flight.")
    col1, col2 = st.columns(2)
    with col1:
        hour = st.slider("Scheduled Departure Hour", 5, 12, 9)
        to_dest = st.selectbox("Destination", options=sorted(df['To'].unique()))
        aircraft = st.selectbox("Aircraft", options=sorted(df['Aircraft'].unique()))
    with col2:
        # Make prediction using the enhanced function
        prediction_value, model_type = make_prediction(hour, to_dest, aircraft, model_data, df)
        
        st.metric(label="Predicted Departure Delay", value=f"{prediction_value:.2f} minutes")
        
        # Show model type info
        if model_type == "ensemble":
            st.success("✨ Enhanced ensemble prediction", icon="🚀")
        elif model_type == "legacy":  
            st.info("📊 Legacy model prediction", icon="ℹ️")
        else:
            st.warning("⚠️ Default estimation", icon="⚠️")
        
        # Prediction interpretation
        if prediction_value < 5:
            st.success("🟢 Excellent time slot - Minimal delay expected")
        elif prediction_value < 15:
            st.warning("🟡 Good time slot - Low delay expected") 
        elif prediction_value < 30:
            st.warning("🟠 Fair time slot - Moderate delay expected")
        else:
            st.error("🔴 Poor time slot - High delay expected")
            
        st.info("💡 This prediction is based on historical patterns. Consider trying different hours to find optimal slots.", icon="ℹ️")

with tab3:
    st.header("Critical Flights Analysis")
    st.markdown("These are the top 10 flights most likely to cause cascading delays if they are disrupted. Prioritizing these can improve overall airport efficiency.")
    st.dataframe(df_critical[['Flight Number', 'Aircraft', 'To', 'STD_datetime', 'centrality']], use_container_width=True)

with tab4:
    st.header("Ask Gemini about the Flight Data")
    st.markdown("Use natural language to ask questions about the flight dataset.")
    user_question = st.text_input("Your question:")
    if user_question:
        with st.spinner("Gemini is analyzing the data..."):
            st.markdown(get_gemini_response(user_question, df))

with tab5:
    st.header("🔬 Model Performance & Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Accuracy Metrics")
        
        # Load model performance metrics
        metrics_path = os.path.join('models', 'model_metrics.json')
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            st.metric("Mean Absolute Error", f"{metrics.get('mae', 0):.2f} minutes", 
                     help="Average prediction error in minutes")
            st.metric("Root Mean Square Error", f"{metrics.get('rmse', 0):.2f} minutes",
                     help="Standard deviation of prediction errors")
            st.metric("R² Score", f"{metrics.get('r2', 0):.3f}",
                     help="Proportion of variance explained by the model (0-1)")
            
            # Training info
            st.markdown("### 📋 Training Information")
            st.info(f"""
            - **Training Samples**: {metrics.get('training_samples', 'N/A'):,}
            - **Test Samples**: {metrics.get('test_samples', 'N/A'):,}
            - **Features Used**: {metrics.get('features_count', 'N/A')}
            """)
        else:
            st.warning("Model metrics not found. Please retrain the model with enhanced features.")
        
        st.subheader("Data Quality Insights")
        
        # Data quality metrics
        st.metric("Total Flight Records", f"{len(df):,}")
        
        # Calculate date range properly
        try:
            date_range = (pd.to_datetime(df['Date']).max() - pd.to_datetime(df['Date']).min()).days
            st.metric("Date Range", f"{date_range} days")
        except:
            st.metric("Date Range", "N/A")
            
        st.metric("Unique Aircraft", f"{df['Aircraft'].nunique()}")
        st.metric("Unique Destinations", f"{df['To'].nunique()}")
    
    with col2:
        st.subheader("Feature Importance")
        
        # Display feature importance
        feature_imp_path = os.path.join('models', 'feature_importance.csv')
        if os.path.exists(feature_imp_path):
            feature_imp = pd.read_csv(feature_imp_path)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            top_features = feature_imp.head(10)
            bars = ax.barh(range(len(top_features)), top_features['importance'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Importance Score')
            ax.set_title('Top 10 Most Important Features')
            ax.invert_yaxis()
            
            # Color bars by importance
            colors = plt.cm.viridis([x/max(top_features['importance']) for x in top_features['importance']])
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            st.pyplot(fig)
        else:
            st.info("Feature importance data not available. Retrain model to see feature analysis.")
        
        st.subheader("Model Recommendations")
        
        # Model performance insights
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                
            r2_score = metrics.get('r2', 0)
            mae_score = metrics.get('mae', 100)
            
            if r2_score > 0.8:
                st.success("🎯 Excellent model performance!")
            elif r2_score > 0.6:
                st.warning("⚠️ Good model performance with room for improvement")
            else:
                st.error("❌ Model needs improvement")
                
            if mae_score < 10:
                st.success("✅ High prediction accuracy (±10 minutes)")
            elif mae_score < 20:
                st.warning("⚠️ Moderate accuracy (±20 minutes)")
            else:
                st.error("❌ Low accuracy - consider more features")
        
        # Recommendations
        st.markdown("### 🎯 Enhancement Recommendations:")
        st.markdown("""
        ✅ **Implemented Enhancements:**
        - Advanced ensemble modeling (RF + GB + XGB)
        - 15+ engineered features
        - Multiple centrality measures for critical flights
        - Enhanced time-based and operational features
        
        🔄 **Future Improvements:**
        - Weather data integration
        - Real-time airport congestion
        - Passenger load factors
        - Seasonal pattern analysis
        - Gate assignment optimization
        """)
        
    # Enhanced Critical Flights Analysis
    st.markdown("---")
    st.subheader("🚨 Enhanced Critical Flights Analysis")
    
    enhanced_critical_path = os.path.join('data', 'enhanced_critical_flights.csv')
    if os.path.exists(enhanced_critical_path):
        enhanced_critical = pd.read_csv(enhanced_critical_path)
        
        if len(enhanced_critical) > 0:
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("**Critical Flights by Centrality Type**")
                
                # Show different centrality measures
                display_df = enhanced_critical[['Flight Number', 'Date', 'combined_centrality', 
                                              'degree_centrality', 'betweenness_centrality', 
                                              'pagerank_score']].head(10)
                st.dataframe(display_df, use_container_width=True)
            
            with col4:
                st.markdown("**Network Connections**")
                
                # Show connection details
                connection_df = enhanced_critical[['Flight Number', 'Date', 'connections_out', 
                                                 'connections_in', 'combined_centrality']].head(10)
                
                # Create a scatter plot of connections
                fig, ax = plt.subplots(figsize=(8, 6))
                scatter = ax.scatter(connection_df['connections_out'], 
                                   connection_df['connections_in'],
                                   s=connection_df['combined_centrality']*1000,
                                   c=connection_df['combined_centrality'], 
                                   cmap='Reds', alpha=0.6)
                ax.set_xlabel('Outgoing Connections')
                ax.set_ylabel('Incoming Connections')
                ax.set_title('Flight Network Connections')
                plt.colorbar(scatter, label='Centrality Score')
                
                # Add annotations for top 3 flights
                for i, row in connection_df.head(3).iterrows():
                    ax.annotate(f"{row['Flight Number']}", 
                              (row['connections_out'], row['connections_in']),
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                st.pyplot(fig)
        else:
            st.info("No enhanced critical flights data available. Please retrain the analysis.")
    else:
        st.info("Enhanced critical flights analysis not found. Retrain models to see detailed network analysis.")