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
    model_path = os.path.join('models', 'delay_predictor.joblib')

    if not os.path.exists(model_path):
        st.error("Model file not found! Please run the backend pipeline first by executing 'python -m src.main' in your terminal.", icon="🚨")
        return None

    model = joblib.load(model_path)
    return model

df, df_critical = load_data()
model = load_model()

def get_gemini_response(question, df):
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

st.title("✈️ AI-Powered Flight Scheduling Assistant")
st.markdown("An interactive dashboard to analyze flight data, predict delays, and identify critical flights at Mumbai Airport (BOM).")

if df is None or df_critical is None or model is None:
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Airport Overview", "⚙️ Schedule Tuner", "🚨 Critical Flights", "💬 Ask Gemini"])

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
        input_df = pd.DataFrame({
            'hour': [hour],
            'day_of_week': [0], 
            'Aircraft_Type': [aircraft.split('(')[0].strip()],
            'To': [to_dest],
            'Aircraft': [aircraft] 
        })
        prediction = model.predict(input_df)
        st.metric(label="Predicted Departure Delay", value=f"{prediction[0]:.2f} minutes")
        st.info("This prediction is based on historical data. A lower value suggests a more optimal time slot.", icon="ℹ️")

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