import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .stAlert {
        padding: 15px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/best_fraud_model.pkl')
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please ensure 'models/best_fraud_model.pkl' exists.")
        return None

@st.cache_data
def load_data():
    try:
        train_df = pd.read_csv('data/processed/fraudTrain.csv')
        test_df = pd.read_csv('data/processed/fraudTest.csv')
        return train_df, test_df
    except FileNotFoundError:
        st.warning("⚠️ Data files not found. Some features will be limited.")
        return None, None

@st.cache_data
def load_comparison():
    try:
        return pd.read_csv('models/model_comparison.csv')
    except FileNotFoundError:
        return None

# Initialize
model = load_model()
train_df, test_df = load_data()
comparison_df = load_comparison()

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/security-shield-green.png", width=100)
st.sidebar.title("🔒 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["🏠 Home", "📊 Model Performance", "🔍 Make Prediction", "📈 Data Analysis", "ℹ️ About"]
)

# Main content
if page == "🏠 Home":
    st.markdown('<p class="main-header">🔒 Credit Card Fraud Detection System</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the Fraud Detection Dashboard
    
    This application uses advanced Machine Learning algorithms to detect fraudulent credit card transactions 
    in real-time. Our system achieves high accuracy while maintaining low false positive rates.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🎯 Model Type", "Random Forest")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if comparison_df is not None:
            best_precision = comparison_df.loc[comparison_df['Precision'].idxmax(), 'Precision']
            st.metric("📊 Best Precision", f"{best_precision:.1%}")
        else:
            st.metric("📊 Precision", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if comparison_df is not None:
            best_recall = comparison_df.loc[comparison_df['Recall'].idxmax(), 'Recall']
            st.metric("🎯 Best Recall", f"{best_recall:.1%}")
        else:
            st.metric("🎯 Recall", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Features
    st.subheader("✨ Key Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - ✅ **Real-time Fraud Detection**
        - ✅ **High Accuracy Models**
        - ✅ **Interactive Visualizations**
        """)
    
    with col2:
        st.markdown("""
        - ✅ **Model Performance Metrics**
        - ✅ **Batch Predictions**
        - ✅ **Data Analysis Tools**
        """)
    
    # Quick Stats
    if train_df is not None:
        st.markdown("---")
        st.subheader("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", f"{len(train_df) + len(test_df):,}")
        with col2:
            fraud_rate = (train_df['is_fraud'].sum() + test_df['is_fraud'].sum()) / (len(train_df) + len(test_df))
            st.metric("Fraud Rate", f"{fraud_rate:.2%}")
        with col3:
            st.metric("Training Samples", f"{len(train_df):,}")
        with col4:
            st.metric("Test Samples", f"{len(test_df):,}")

elif page == "📊 Model Performance":
    st.markdown('<p class="main-header">📊 Model Performance Analysis</p>', unsafe_allow_html=True)
    
    if comparison_df is not None:
        # Model Comparison
        st.subheader("🔍 Model Comparison")
        
        # Display comparison table
        st.dataframe(
            comparison_df.style.highlight_max(axis=0, subset=['Precision', 'Recall', 'F1-Score', 'Accuracy'])
            .format({'Precision': '{:.3f}', 'Recall': '{:.3f}', 'F1-Score': '{:.3f}', 'Accuracy': '{:.3f}'})
        )
        
        # Metrics comparison chart
        st.subheader("📈 Metrics Visualization")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Precision', 'Recall', 'F1-Score', 'Accuracy'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
        positions = [(1,1), (1,2), (2,1), (2,2)]
        
        for metric, pos in zip(metrics, positions):
            fig.add_trace(
                go.Bar(
                    x=comparison_df['Model'],
                    y=comparison_df[metric],
                    name=metric,
                    text=comparison_df[metric].apply(lambda x: f'{x:.3f}'),
                    textposition='auto',
                ),
                row=pos[0], col=pos[1]
            )
        
        fig.update_layout(height=800, showlegend=False, title_text="Model Performance Metrics")
        st.plotly_chart(fig, use_container_width=True)
        
        # Best Model Highlight
        st.markdown("---")
        best_model = comparison_df.loc[comparison_df['F1-Score'].idxmax()]
        
        st.success(f"""
        ### 🏆 Best Performing Model: **{best_model['Model']}**
        
        - **Precision**: {best_model['Precision']:.3f}
        - **Recall**: {best_model['Recall']:.3f}
        - **F1-Score**: {best_model['F1-Score']:.3f}
        - **Accuracy**: {best_model['Accuracy']:.3f}
        """)
    else:
        st.warning("Model comparison data not found.")

elif page == "🔍 Make Prediction":
    st.markdown('<p class="main-header">🔍 Fraud Detection Prediction</p>', unsafe_allow_html=True)
    
    if model is None:
        st.error("Model not loaded. Please check if the model file exists.")
    else:
        tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
        
        with tab1:
            st.subheader("Enter Transaction Details")
            
            # Create feature columns based on your processed data
            col1, col2 = st.columns(2)
            
            with col1:
                amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=10.0)
                hour = st.slider("Hour of Day", 0, 23, 12)
                day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 3)
                merchant_encoded = st.number_input("Merchant ID (encoded)", min_value=0, value=0)
                category_encoded = st.number_input("Category ID (encoded)", min_value=0, value=0)
            
            with col2:
                age = st.slider("Cardholder Age", 18, 100, 35)
                city_pop = st.number_input("City Population", min_value=0, value=50000)
                amt_log = st.number_input("Amount (log scale)", min_value=0.0, value=4.6, step=0.1)
                dist_from_home = st.number_input("Distance from Home (km)", min_value=0.0, value=10.0)
                dist_from_last = st.number_input("Distance from Last Transaction (km)", min_value=0.0, value=5.0)
            
            if st.button("🔍 Predict Fraud", type="primary"):
                # Create feature array (adjust based on your model's features)
                features = np.array([[amt, hour, day_of_week, merchant_encoded, category_encoded, 
                                    age, city_pop, amt_log, dist_from_home, dist_from_last]])
                
                try:
                    prediction = model.predict(features)[0]
                    probability = model.predict_proba(features)[0]
                    
                    st.markdown("---")
                    
                    if prediction == 1:
                        st.error("⚠️ **FRAUDULENT TRANSACTION DETECTED!**")
                        st.error(f"Fraud Probability: {probability[1]:.2%}")
                    else:
                        st.success("✅ **LEGITIMATE TRANSACTION**")
                        st.success(f"Fraud Probability: {probability[1]:.2%}")
                    
                    # Probability visualization
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=probability[1] * 100,
                        title={'text': "Fraud Risk Score"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "red" if prediction == 1 else "green"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
        
        with tab2:
            st.subheader("Upload CSV File for Batch Prediction")
            
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    st.write("Preview of uploaded data:")
                    st.dataframe(batch_df.head())
                    
                    if st.button("🔍 Run Batch Prediction"):
                        with st.spinner("Processing..."):
                            # Remove target column if present
                            X_batch = batch_df.drop('is_fraud', axis=1, errors='ignore')
                            
                            predictions = model.predict(X_batch)
                            probabilities = model.predict_proba(X_batch)[:, 1]
                            
                            batch_df['Prediction'] = predictions
                            batch_df['Fraud_Probability'] = probabilities
                            batch_df['Risk_Level'] = pd.cut(
                                probabilities, 
                                bins=[0, 0.3, 0.7, 1.0],
                                labels=['Low', 'Medium', 'High']
                            )
                            
                            st.success("✅ Predictions completed!")
                            
                            # Summary stats
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Transactions", len(batch_df))
                            with col2:
                                fraud_count = predictions.sum()
                                st.metric("Fraudulent", fraud_count)
                            with col3:
                                fraud_pct = (fraud_count / len(batch_df)) * 100
                                st.metric("Fraud Rate", f"{fraud_pct:.2f}%")
                            
                            st.dataframe(batch_df)
                            
                            # Download results
                            csv = batch_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name=f"fraud_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

elif page == "📈 Data Analysis":
    st.markdown('<p class="main-header">📈 Data Analysis & Insights</p>', unsafe_allow_html=True)
    
    if train_df is not None and test_df is not None:
        # Combine datasets
        full_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # Fraud distribution
        st.subheader("📊 Fraud Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fraud_counts = full_df['is_fraud'].value_counts()
            fig = px.pie(
                values=fraud_counts.values,
                names=['Legitimate', 'Fraudulent'],
                title='Transaction Distribution',
                color_discrete_sequence=['#00CC96', '#EF553B']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                x=['Legitimate', 'Fraudulent'],
                y=fraud_counts.values,
                title='Transaction Count by Type',
                labels={'x': 'Transaction Type', 'y': 'Count'},
                color=['Legitimate', 'Fraudulent'],
                color_discrete_sequence=['#00CC96', '#EF553B']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature analysis
        st.subheader("📊 Feature Analysis")
        
        numeric_cols = full_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'is_fraud']
        
        selected_feature = st.selectbox("Select Feature to Analyze", numeric_cols)
        
        if selected_feature:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    full_df,
                    x=selected_feature,
                    color='is_fraud',
                    title=f'{selected_feature} Distribution',
                    labels={'is_fraud': 'Transaction Type'},
                    color_discrete_map={0: '#00CC96', 1: '#EF553B'},
                    barmode='overlay',
                    opacity=0.7
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(
                    full_df,
                    x='is_fraud',
                    y=selected_feature,
                    title=f'{selected_feature} by Transaction Type',
                    labels={'is_fraud': 'Transaction Type'},
                    color='is_fraud',
                    color_discrete_map={0: '#00CC96', 1: '#EF553B'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("🔥 Feature Correlation")
        
        corr_cols = st.multiselect(
            "Select features for correlation analysis",
            numeric_cols,
            default=numeric_cols[:5]
        )
        
        if len(corr_cols) >= 2:
            corr_matrix = full_df[corr_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect='auto',
                title='Feature Correlation Matrix',
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Data files not found. Please ensure data is available.")

else:  # About page
    st.markdown('<p class="main-header">ℹ️ About This Application</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🔒 Credit Card Fraud Detection System
    
    ### Overview
    This application leverages Machine Learning to detect fraudulent credit card transactions in real-time. 
    Built with state-of-the-art algorithms, it provides high accuracy while maintaining low false positive rates.
    
    ### 🎯 Features
    
    - **Real-time Prediction**: Instant fraud detection for individual transactions
    - **Batch Processing**: Analyze multiple transactions at once
    - **Model Comparison**: Compare performance of different ML algorithms
    - **Interactive Visualizations**: Explore data patterns and insights
    - **Comprehensive Metrics**: Precision, Recall, F1-Score, and Accuracy
    
    ### 🤖 Models Used
    
    - **Logistic Regression**: Baseline linear model
    - **Decision Tree**: Tree-based classification
    - **Random Forest**: Ensemble learning (Best performer)
    - **XGBoost**: Gradient boosting algorithm
    
    ### 📊 Dataset
    
    The system is trained on a comprehensive credit card transaction dataset with features including:
    - Transaction amount
    - Time-based features
    - Merchant information
    - Geographic data
    - User demographics
    
    ### 🛠️ Technology Stack
    
    - **Machine Learning**: Scikit-learn, XGBoost
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Plotly, Seaborn
    - **Web Framework**: Streamlit
    
    ### 📈 Performance
    
    Our best model (Random Forest) achieves:
    - High precision in detecting fraud
    - Excellent recall to minimize missed frauds
    - Balanced F1-score for overall performance
    - Low false positive rate
    
    ### 👥 Contact & Support
    
    For questions or support, please contact the development team.
    
    ---
    
    *Last Updated: 2025*
    """)
    
    # Model info
    if comparison_df is not None:
        st.markdown("### 📊 Current Model Statistics")
        best_model_stats = comparison_df.iloc[comparison_df['F1-Score'].idxmax()]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model", best_model_stats['Model'])
        with col2:
            st.metric("Precision", f"{best_model_stats['Precision']:.3f}")
        with col3:
            st.metric("Recall", f"{best_model_stats['Recall']:.3f}")
        with col4:
            st.metric("F1-Score", f"{best_model_stats['F1-Score']:.3f}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    🔒 Fraud Detection System | Built with Streamlit | © 2025
</div>
""", unsafe_allow_html=True)