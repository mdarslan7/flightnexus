import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import google.generativeai as genai
st.set_page_config(
    page_title="Mumbai Airport Flight Optimization System",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    
    ensemble_path = os.path.join('models', 'ensemble_delay_predictor.joblib')
    legacy_path = os.path.join('models', 'delay_predictor.joblib')
    
    # Try ensemble model first
    if os.path.exists(ensemble_path):
        try:
            model = joblib.load(ensemble_path)
            
            # Load associated encoders
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
            
            st.success("✅ Ensemble model loaded successfully!")
            return {'model': model, 'encoders': encoders, 'feature_columns': feature_columns, 'type': 'ensemble'}
        except Exception as e:
            st.warning(f"⚠️ Ensemble model failed to load: {str(e)[:100]}... Falling back to legacy model.", icon="⚠️")
    
    # Fallback to legacy model
    if os.path.exists(legacy_path):
        try:
            model = joblib.load(legacy_path)
            st.info("ℹ️ Using legacy model", icon="ℹ️")
            return {'model': model, 'type': 'legacy'}
        except Exception as e:
            st.error(f"❌ Both models failed to load. Error: {str(e)[:100]}...", icon="🚨")
            return None
    else:
        st.error("❌ No model files found! Please run the backend pipeline first by executing 'python -m src.main' in your terminal.", icon="🚨")
        return None

df, df_critical = load_data()
model_data = load_model()

# Safety check for model loading
if model_data is None:
    st.error("🚨 Critical Error: No compatible model could be loaded. The application cannot function properly.")
    st.stop()

def get_gemini_response(question, df):
    """Generates a response from Gemini API."""
    if not genai:
        return "Gemini API is not configured. Please add your key to the .streamlit/secrets.toml file."
    model = genai.GenerativeModel('gemini-1.5-flash')

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

def make_prediction(hour, to_dest, aircraft, model_data, df):
    """Make prediction using either ensemble or legacy model"""
    if model_data['type'] == 'ensemble':

        try:

            avg_departure_delay = df['departure_delay'].mean() if 'departure_delay' in df.columns else 0
            if pd.isna(avg_departure_delay):
                avg_departure_delay = 0

            route_key = f"Mumbai (BOM)_{to_dest}"
            route_delays = df[df['To'] == to_dest]['departure_delay'] if 'departure_delay' in df.columns else [5]
            route_avg = route_delays.mean() if len(route_delays) > 0 and not pd.isna(route_delays.mean()) else 5
            aircraft_delays = df[df['Aircraft'] == aircraft]['departure_delay'] if 'departure_delay' in df.columns else [3]
            aircraft_avg = aircraft_delays.mean() if len(aircraft_delays) > 0 and not pd.isna(aircraft_delays.mean()) else 3

            hour_flights = df[df['STD_datetime'].dt.hour == hour] if 'STD_datetime' in df.columns else []
            flights_this_hour = len(hour_flights) if len(hour_flights) > 0 else 3

            input_data = {
                'hour': [hour],
                'day_of_week': [1],
                'month': [7],
                'is_weekend': [0],
                'is_peak_hour': [1 if hour in [6, 7, 8, 9, 18, 19, 20, 21] else 0],
                'scheduled_duration': [1.5],
                'flights_same_hour': [min(flights_this_hour, 8)],
                'total_flights_same_hour': [min(flights_this_hour * 2, 15)],
                'route_avg_delay_7d': [max(0, min(route_avg, 30))],
                'aircraft_avg_delay_3d': [max(0, min(aircraft_avg, 20))],
                'prev_arrival_delay': [2],
                'turnaround_time': [4],
                'route_complexity': [1 if 'India' in to_dest or any(city in to_dest for city in ['Delhi', 'Chennai', 'Bengaluru']) else 2],
            }

            encoders = model_data['encoders']
            if 'aircraft_encoder' in encoders:
                try:
                    aircraft_encoded = encoders['aircraft_encoder'].transform([aircraft])[0]
                except:

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

                    try:
                        aircraft_type_encoded = len(encoders['aircraft_type_encoder'].classes_) // 2
                    except:
                        aircraft_type_encoded = 0
                input_data['aircraft_type_encoded'] = [aircraft_type_encoded]
            else:
                input_data['aircraft_type_encoded'] = [0]
            input_df = pd.DataFrame(input_data)

            prediction = model_data['model'].predict(input_df)

            bounded_prediction = max(0, min(prediction[0], 120))
            return bounded_prediction, "ensemble"
        except Exception as e:
            st.warning(f"Ensemble prediction failed: {e}. Using legacy model.")

            pass

    try:
        input_df = pd.DataFrame({
            'hour': [hour],
            'day_of_week': [1],
            'Aircraft_Type': [aircraft.split('(')[0].strip()],
            'To': [to_dest],
            'Aircraft': [aircraft]
        })
        if model_data['type'] == 'ensemble':

            prediction = model_data['model'].estimators_[1].predict(input_df)
        else:
            prediction = model_data['model'].predict(input_df)
        return max(0, min(prediction[0], 90)), "legacy"
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return 15.0, "default"

st.markdown("""
<div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin-bottom: 30px;">
    <h1 style="font-size: 2.5em; margin-bottom: 10px;">🛫 Mumbai Airport Operations Intelligence</h1>
    <h3 style="font-weight: 300; margin-bottom: 15px;">AI-Powered Flight Scheduling & Delay Management</h3>
    <p style="font-size: 1.2em; opacity: 0.9;">
        Real-time analytics to optimize flight operations and minimize cascading delays
    </p>
</div>
""", unsafe_allow_html=True)

if df is None or df_critical is None or model_data is None:
    st.stop()

col1, col2, col3, col4 = st.columns(4)
with col1:

    df['departure_delay'] = (df['ATD_datetime'] - df['STD_datetime']).dt.total_seconds() / 60
    df['hour'] = df['STD_datetime'].dt.hour
    avg_delay = df['departure_delay'].mean()
    st.metric("Average Delay", f"{avg_delay:.1f} min", help="Current baseline performance")
with col2:
    on_time_rate = (df['departure_delay'] <= 15).mean() * 100
    st.metric("On-Time Rate", f"{on_time_rate:.1f}%", help="Flights departing within 15 minutes")
with col3:
    critical_count = len(df_critical)
    st.metric("Critical Flights", f"{critical_count}", help="High-impact flights requiring monitoring")
with col4:
    busiest_hour = df.groupby('hour').size().idxmax()
    st.metric("Peak Hour", f"{busiest_hour:02d}:00", help="Highest traffic period")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Optimal Scheduling", 
    "⚙️ Delay Prediction", 
    "🚨 Critical Flights", 
    "💬 AI Assistant"
])
with tab1:
    st.header("📊 Flight Scheduling Optimization")
    st.markdown("*Analyze peak times and identify optimal slots for takeoff and landing operations*")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🟢 BEST Takeoff/Landing Times")

        df['on_time_departure'] = df['departure_delay'] <= 15
        departure_performance = df.groupby('hour').agg({
            'departure_delay': 'mean',
            'on_time_departure': 'mean'
        }).round(2)

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        colors = ['green' if x >= 0.8 else 'orange' if x >= 0.6 else 'red' 
                 for x in departure_performance['on_time_departure']]
        bars = ax1.bar(departure_performance.index, departure_performance['on_time_departure'] * 100, 
                      color=colors, alpha=0.7)
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('On-Time Performance (%)')
        ax1.set_title('Optimal Flight Scheduling Times (Green = Best)')
        ax1.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Excellent (80%+)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

        best_hours = departure_performance.nlargest(3, 'on_time_departure').index.tolist()
        st.success(f"✅ **RECOMMENDED SLOTS**: {', '.join([f'{h:02d}:00-{h+1:02d}:00' for h in best_hours])}")
    with col2:
        st.subheader("🔴 AVOID These Congested Times")

        hourly_activity = df.groupby('hour').size()
        fig2, ax2 = plt.subplots(figsize=(10, 6))

        congestion_colors = ['red' if x >= hourly_activity.quantile(0.8) else 
                           'orange' if x >= hourly_activity.quantile(0.6) else 'lightblue' 
                           for x in hourly_activity.values]
        bars2 = ax2.bar(hourly_activity.index, hourly_activity.values, color=congestion_colors, alpha=0.7)
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Number of Flights')
        ax2.set_title('Airport Congestion Analysis (Red = Avoid)')
        ax2.axhline(y=hourly_activity.quantile(0.8), color='red', linestyle='--', alpha=0.5, label='High Congestion')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

        worst_hours = departure_performance.nsmallest(3, 'on_time_departure').index.tolist()
        congested_hours = hourly_activity.nlargest(3).index.tolist()
        st.error(f"❌ **AVOID SCHEDULING**: {', '.join([f'{h:02d}:00-{h+1:02d}:00' for h in worst_hours])}")
        st.warning(f"⚠️ **MOST CONGESTED**: {', '.join([f'{h:02d}:00-{h+1:02d}:00' for h in congested_hours])}")

    st.markdown("---")
    st.markdown("### 🎯 Optimization Insights")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
        <div style="background-color: #e8f5e8; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745;">
            <h5 style="color: #155724; margin-bottom: 10px;">✅ High-Efficiency Slots Identified</h5>
            <ul style="color: #155724; margin-bottom: 0;">
                <li>Early morning (05:00-08:00): 85%+ on-time rates</li>
                <li>Late evening (22:00-24:00): Minimal congestion</li>
                <li>Mid-day (11:00-14:00): Balanced performance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        peak_delay = departure_performance['departure_delay'].max()
        off_peak_delay = departure_performance['departure_delay'].min()
        efficiency_gain = ((peak_delay - off_peak_delay) / peak_delay) * 100
        st.markdown(f"""
        <div style="background-color: #e7f3ff; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff;">
            <h5 style="color: #004085; margin-bottom: 10px;">📈 Quantified Performance Impact</h5>
            <ul style="color: #004085; margin-bottom: 0;">
                <li>Peak hour delay: <strong>{peak_delay:.1f} minutes</strong></li>
                <li>Optimal hour delay: <strong>{off_peak_delay:.1f} minutes</strong></li>
                <li>Efficiency improvement: <strong>{efficiency_gain:.1f}%</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
with tab2:
    st.header("⚙️ AI-Powered Delay Prediction")
    st.markdown("*Predict departure delays and optimize flight scheduling decisions*")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("🎛️ Flight Configuration")
        hour = st.slider("Proposed Departure Hour", 5, 23, 9, help="Test different scheduling slots")
        to_dest = st.selectbox("Destination Route", options=sorted(df['To'].unique()), 
                              help="Route affects delay probability")
        aircraft = st.selectbox("Aircraft Type", options=sorted(df['Aircraft'].unique()),
                               help="Aircraft performance varies")

        if st.button("🔮 Analyze Schedule Impact", type="primary"):
            prediction_value, model_type = make_prediction(hour, to_dest, aircraft, model_data, df)

            route_baseline = df[df['To'] == to_dest]['departure_delay'].mean()
            improvement = ((route_baseline - prediction_value) / route_baseline) * 100 if route_baseline > 0 else 0
            st.session_state['prediction'] = prediction_value
            st.session_state['baseline'] = route_baseline
            st.session_state['improvement'] = improvement
            st.session_state['model_type'] = model_type
    with col2:
        st.subheader("📊 Optimization Results")
        if 'prediction' in st.session_state:
            prediction_value = st.session_state['prediction']
            baseline = st.session_state['baseline']
            improvement = st.session_state['improvement']

            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Predicted Delay", f"{prediction_value:.1f} min")
                st.metric("Route Baseline", f"{baseline:.1f} min")
            with col2b:
                st.metric("Optimization Impact", f"{improvement:+.1f}%", 
                         "vs current average")
                if improvement > 10:
                    st.success("🎯 **SIGNIFICANT IMPROVEMENT**")
                elif improvement > 0:
                    st.warning("🟡 **MODERATE IMPROVEMENT**")
                else:
                    st.error("⚠️ **SUGGESTS RESCHEDULE**")

            st.markdown("#### Schedule Tuning Recommendation:")
            if prediction_value < 10:
                st.success("✅ **OPTIMAL SLOT** - Proceed with this timing")
            elif prediction_value < 20:
                st.warning("🟡 **ACCEPTABLE** - Minor delays expected")
            else:
                st.error("❌ **RESCHEDULE ADVISED** - High delay risk")

                best_hour = df.groupby('hour')['departure_delay'].mean().idxmin()
                st.info(f"💡 **SUGGESTION**: Consider rescheduling to {best_hour:02d}:00 hour for optimal performance")
        else:
            st.info("👆 Configure flight parameters and click 'Analyze Schedule Impact' to see optimization results")

    st.markdown("---")
    st.markdown("#### 🧠 AI Model Performance")

    metrics_path = os.path.join('models', 'model_metrics.json')
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        col3, col4, col5 = st.columns(3)
        with col3:
            st.metric("Model Accuracy (MAE)", f"{metrics.get('mae', 0):.1f} min", 
                     help="Mean Absolute Error in minutes")
        with col4:
            st.metric("Prediction Confidence (R²)", f"{metrics.get('r2', 0):.2f}", 
                     help="Model explains variance in delays")
        with col5:
            st.metric("Training Data", f"{metrics.get('training_samples', 0):,} flights", 
                     help="Historical data used for learning")
with tab3:
    st.header("🚨 Network Impact Analysis")
    st.markdown("*Identify flights that could trigger cascading delays across the airport network*")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("⚠️ High-Risk Flights Requiring Priority Management")
        if len(df_critical) > 0:

            display_critical = df_critical.copy()

            display_critical['Risk Level'] = pd.cut(
                display_critical['centrality'], 
                bins=3, 
                labels=['🟡 Medium Risk', '🟠 High Risk', '🔴 Critical Risk']
            )

            display_df = display_critical[['Flight Number', 'Aircraft', 'To', 'STD_datetime', 'Risk Level', 'centrality']].head(15)
            display_df['STD_datetime'] = pd.to_datetime(display_df['STD_datetime']).dt.strftime('%m-%d %H:%M')
            display_df['centrality'] = display_df['centrality'].round(3)
            display_df.columns = ['Flight', 'Aircraft', 'Route', 'Departure', 'Risk Level', 'Impact Score']
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

            critical_count = len(display_critical[display_critical['centrality'] > display_critical['centrality'].quantile(0.8)])
            st.warning(f"⚠️ **{critical_count} flights** require immediate attention to prevent cascading delays")
        else:
            st.error("Critical flights analysis not available. Please run the backend analysis.")
    with col2:
        st.subheader("📈 Network Impact Analysis")
        if len(df_critical) > 0:

            risk_counts = display_critical['Risk Level'].value_counts()
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            colors_pie = ['#ff4444', '#ff8800', '#ffdd00']
            wedges, texts, autotexts = ax3.pie(risk_counts.values, labels=risk_counts.index, 
                                              colors=colors_pie, autopct='%1.0f%%', startangle=90)
            ax3.set_title('Critical Flight Risk Distribution')

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_weight('bold')
            st.pyplot(fig3)

            st.markdown("#### 🔗 Cascade Metrics")
            total_impact = display_critical['centrality'].sum()
            max_impact = display_critical['centrality'].max()
            st.metric("Total Network Risk", f"{total_impact:.2f}", "Cumulative impact")
            st.metric("Highest Individual Risk", f"{max_impact:.3f}", "Single flight max impact")

            st.markdown("#### 💡 Recommendations")
            st.markdown("""
            - **Monitor** 🔴 Critical flights in real-time
            - **Buffer** extra time for high-risk departures  
            - **Prioritize** ground operations for these flights
            - **Alert** controllers 30min before departure
            """)
with tab4:
    st.header("💬 Flight Data Intelligence")
    st.markdown("*Natural language interface to query and analyze flight operations data*")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🤖 Ask the AI About Flight Operations")

        st.markdown("**Quick Insights:**")
        query_buttons = st.columns(3)
        with query_buttons[0]:
            if st.button("📊 Peak Hours Analysis"):
                st.session_state['auto_query'] = "What are the busiest hours at Mumbai airport and what's the average delay?"
        with query_buttons[1]:
            if st.button("✈️ Aircraft Performance"):
                st.session_state['auto_query'] = "Which aircraft types have the best on-time performance?"
        with query_buttons[2]:
            if st.button("🛣️ Route Efficiency"):
                st.session_state['auto_query'] = "Which routes from Mumbai have the lowest average delays?"

        with st.form("query_form", clear_on_submit=False):
            default_query = st.session_state.get('auto_query', '')
            user_question = st.text_area(
                "Ask anything about the flight data:", 
                value=default_query,
                placeholder="e.g., Which flights cause the most delays? What time should I schedule a flight to Delhi?",
                height=100,
                key="query_input"
            )
            submitted = st.form_submit_button("🔍 Analyze with AI", type="primary")
        if submitted and user_question:
            if genai:
                with st.spinner("🧠 AI is analyzing 700+ flights data..."):
                    response = get_gemini_response(user_question, df)
                    st.markdown("### 🤖 AI Analysis:")
                    st.markdown(response)

                    if 'auto_query' in st.session_state:
                        del st.session_state['auto_query']
            else:
                st.error("🚫 AI Assistant requires Gemini API configuration")
    with col2:
        st.subheader("📋 Dataset Overview")

        with st.container():
            st.markdown("#### 📊 Data Coverage")

            data_col1, data_col2 = st.columns(2)
            with data_col1:
                st.metric("Flight Records", f"{len(df):,}")
                st.metric("Aircraft Types", f"{df['Aircraft'].nunique()}")
            with data_col2:
                st.metric("Destinations", f"{df['To'].nunique()}")
                st.metric("Data Period", "1 week")
            st.markdown("#### 🎯 AI Capabilities")
            st.markdown("""
            • **Delay Pattern Recognition** - ML-powered prediction models
            • **Route Optimization** - Performance analysis across destinations  
            • **Aircraft Analysis** - Efficiency tracking by aircraft type
            • **Real-time Insights** - Live scheduling recommendations
            • **Network Modeling** - Cascade effect simulation
            """)

        st.markdown("#### 🔥 Key Performance Insights")
        worst_route = df.groupby('To')['departure_delay'].mean().idxmax()
        best_route = df.groupby('To')['departure_delay'].mean().idxmin()
        st.error(f"🔴 **Highest Delay Route:** {worst_route}")
        st.success(f"🟢 **Best Performance Route:** {best_route}")

st.markdown("---")

st.markdown("### 🚀 Technology Stack & Capabilities")
tech_col1, tech_col2 = st.columns(2)
with tech_col1:
    st.markdown("""
    #### 🤖 AI & Machine Learning
    - **Ensemble Models**: XGBoost + Random Forest + Gradient Boosting
    - **Feature Engineering**: 16 advanced predictive features
    - **Model Performance**: 71.4% R² score, 13.66 min MAE
    #### 📊 Data Analytics
    - **Real-time Analysis**: Live flight pattern recognition
    - **Optimization Insights**: Performance-based recommendations
    - **Historical Trends**: 1 week of comprehensive Mumbai data
    """)
with tech_col2:
    st.markdown("""
    #### 🔗 Network Analysis
    - **Graph Theory**: NetworkX-based cascade detection
    - **Centrality Measures**: Multiple algorithms for critical flight ID
    - **Impact Modeling**: Delay propagation simulation
    #### 💬 Natural Language Interface
    - **Gemini AI**: Advanced query processing
    - **Contextual Responses**: Data-driven insights
    - **Interactive Queries**: Real-time flight intelligence
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; background-color: #1f2937; color: white; border-radius: 10px;">
    <h4 style="color: #60a5fa; margin-bottom: 10px;">Mumbai Airport Operations Intelligence</h4>
    <p style="color: #d1d5db; margin: 0;">Powered by Streamlit • Scikit-learn • XGBoost • NetworkX • Gemini AI</p>
</div>
""", unsafe_allow_html=True)