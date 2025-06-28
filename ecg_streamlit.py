import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import io
from datetime import datetime
import warnings
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
warnings.filterwarnings('ignore')

# ================================
# ResNet Architecture (Copy from your main code)
# ================================

class Bottleneck1D(nn.Module):
    """Bottleneck residual block for 1D signals (used in ResNet-50)"""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck1D, self).__init__()
        
        # 1x1 conv for dimensionality reduction
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # 3x3 conv with stride
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 1x1 conv for dimensionality expansion
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, 
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        # 1x1 conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 3x3 conv
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # 1x1 conv
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class ResNet1D(nn.Module):
    """ResNet architecture adapted for 1D ECG signals"""
    
    def __init__(self, block, layers, num_classes=5, input_channels=1):
        super(ResNet1D, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, 
                              padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )
            
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

def resnet50_1d(num_classes=5):
    """Create ResNet-50 for 1D ECG signals"""
    return ResNet1D(Bottleneck1D, [3, 4, 6, 3], num_classes=num_classes)

# ================================
# Streamlit App Configuration
# ================================

def setup_page():
    """Configure Streamlit page"""
    st.set_page_config(
        page_title="ECG Arrhythmia Classification",
        page_icon="🫀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-container {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
        color: #000000 !important;
    }
    .prediction-container.correct {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .prediction-container.incorrect {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .prediction-container strong {
        color: #000000 !important;
    }
    .prediction-text {
        color: #000000 !important;
        font-weight: bold;
    }
    .warning-container {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ================================
# Model Loading Functions
# ================================

@st.cache_resource
def load_saved_model(model_path):
    """Load saved ResNet-50 model"""
    try:
        if os.path.exists(model_path):
            model = torch.load(model_path, map_location='cpu')
            model.eval()
            return model
        else:
            st.error(f"Model file not found: {model_path}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def create_default_model():
    """Create default ResNet-50 model if no saved model available"""
    model = resnet50_1d(num_classes=5)
    model.eval()
    return model

def get_fixed_model_path():
    """Return the fixed trained model path"""
    return "/home/irman/dskc_cnn/saved_models/ResNet-50_20250623_111959/complete_model.pth"

def get_fixed_model_info():
    """Get information about the fixed model"""
    model_dir = "/home/irman/dskc_cnn/saved_models/ResNet-50_20250623_103251"
    try:
        # Try to load summary info
        summary_path = os.path.join(model_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                return json.load(f)
        else:
            # Return default info if summary not found
            return {
                "model_name": "ResNet-50",
                "timestamp": "20250623_103251",
                "test_accuracy": "Trained Model",
                "total_parameters": "Unknown"
            }
    except Exception as e:
        return {
            "model_name": "ResNet-50",
            "timestamp": "20250623_103251", 
            "test_accuracy": "Trained Model",
            "total_parameters": "Unknown"
        }

def find_best_model():
    """Return the fixed model path"""
    return get_fixed_model_path()

# ================================
# Data Processing Functions
# ================================

def validate_csv_data(df, has_labels=False):
    """Validate uploaded CSV data"""
    errors = []
    warnings = []
    
    # Check if DataFrame is empty
    if df.empty:
        errors.append("CSV file is empty")
        return errors, warnings
    
    # Check number of columns based on whether labels are included
    expected_features = 187
    if has_labels:
        expected_total = 188  # 187 features + 1 label
        if df.shape[1] < expected_total:
            errors.append(f"Expected {expected_total} columns (187 features + 1 label), got {df.shape[1]} columns")
        elif df.shape[1] > expected_total:
            warnings.append(f"CSV has {df.shape[1]} columns. Using first 187 for features and last column as labels.")
    else:
        if df.shape[1] < expected_features:
            errors.append(f"Expected at least {expected_features} features, got {df.shape[1]} columns")
        elif df.shape[1] > expected_features:
            warnings.append(f"CSV has {df.shape[1]} columns. Using first {expected_features} for prediction.")
    
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        warnings.append(f"Found {missing_count} missing values. They will be replaced with 0.")
    
    # Check data types for feature columns
    feature_cols = df.iloc[:, :expected_features]
    non_numeric = feature_cols.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        errors.append(f"Non-numeric feature columns found: {list(non_numeric)}")
    
    return errors, warnings

def preprocess_csv_data(df, has_labels=False):
    """Preprocess CSV data for model input"""
    if has_labels:
        # Separate features and labels
        features_df = df.iloc[:, :187]  # First 187 columns as features
        labels = df.iloc[:, -1].values.astype(int)  # Last column as labels
        
        # Take first 187 columns as features
        if features_df.shape[1] < 187:
            # Pad with zeros if less than 187 features
            padding_needed = 187 - features_df.shape[1]
            padding = pd.DataFrame(np.zeros((features_df.shape[0], padding_needed)))
            features_df = pd.concat([features_df, padding], axis=1)
        
        # Replace missing values with 0
        features_df = features_df.fillna(0)
        
        # Convert to numpy array
        data = features_df.values.astype(np.float32)
        
        return data, labels
    else:
        # Take first 187 columns (ECG features)
        if df.shape[1] > 187:
            df = df.iloc[:, :187]
        elif df.shape[1] < 187:
            # Pad with zeros if less than 187 features
            padding_needed = 187 - df.shape[1]
            padding = pd.DataFrame(np.zeros((df.shape[0], padding_needed)))
            df = pd.concat([df, padding], axis=1)
        
        # Replace missing values with 0
        df = df.fillna(0)
        
        # Convert to numpy array
        data = df.values.astype(np.float32)
        
        return data, None

def normalize_data(data):
    """Normalize data (simple min-max normalization)"""
    normalized_data = data.copy()
    for i in range(data.shape[0]):
        row = data[i]
        if np.max(row) != np.min(row):  # Avoid division by zero
            normalized_data[i] = (row - np.min(row)) / (np.max(row) - np.min(row))
    return normalized_data

def predict_ecg_batch(model, data, class_names):
    """Make predictions on batch of ECG signals"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Normalize data
    normalized_data = normalize_data(data)
    
    # Convert to tensor and add channel dimension
    data_tensor = torch.FloatTensor(normalized_data).unsqueeze(1)  # [batch, 1, 187]
    data_tensor = data_tensor.to(device)
    
    predictions = []
    
    with torch.no_grad():
        output = model(data_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_classes = torch.argmax(output, dim=1)
        max_confidences = torch.max(probabilities, dim=1)[0]
        
        for i in range(len(data)):
            pred_class = predicted_classes[i].item()
            confidence = max_confidences[i].item()
            probs = probabilities[i].cpu().numpy()
            
            predictions.append({
                'class_id': pred_class,
                'class_name': class_names[pred_class],
                'confidence': confidence,
                'probabilities': probs
            })
    
    return predictions

def calculate_metrics(predictions, true_labels, class_names):
    """Calculate classification metrics"""
    pred_labels = [p['class_id'] for p in predictions]
    
    accuracy = accuracy_score(true_labels, pred_labels)
    
    # Classification report
    class_report = classification_report(
        true_labels, pred_labels, 
        target_names=[class_names[i] for i in range(len(class_names))],
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    return accuracy, class_report, cm

# ================================
# Visualization Functions
# ================================

def plot_ecg_signal(signal, title="ECG Signal", sample_idx=0, true_label=None, pred_label=None):
    """Plot ECG signal using Plotly"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=signal,
        mode='lines',
        name=f'Sample {sample_idx + 1}',
        line=dict(color='blue', width=2)
    ))
    
    # Update title to include prediction info if available
    title_text = title
    if true_label is not None and pred_label is not None:
        correct = "✓" if true_label == pred_label else "✗"
        title_text = f"{title}<br>True: {true_label} | Pred: {pred_label} {correct}"
    
    fig.update_layout(
        title=title_text,
        xaxis_title='Time Points',
        yaxis_title='Amplitude',
        height=400,
        template='plotly_white'
    )
    
    return fig

def plot_prediction_probabilities(probabilities, class_names, sample_idx=0, true_class_id=None):
    """Plot prediction probabilities as bar chart"""
    classes = [class_names[i] for i in range(len(class_names))]
    
    # Color bars: green for correct prediction, red for incorrect, blue for others
    colors = []
    for i, p in enumerate(probabilities):
        if p == max(probabilities):  # Predicted class
            if true_class_id is not None and i == true_class_id:
                colors.append('green')  # Correct prediction
            elif true_class_id is not None:
                colors.append('red')    # Incorrect prediction
            else:
                colors.append('darkblue')  # No true label available
        elif true_class_id is not None and i == true_class_id:
            colors.append('orange')     # True class but not predicted
        else:
            colors.append('lightblue')  # Other classes
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=probabilities,
            marker_color=colors,
            text=[f'{p:.3f}' for p in probabilities],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=f'Class Prediction Probabilities - Sample {sample_idx + 1}',
        xaxis_title='Arrhythmia Classes',
        yaxis_title='Probability',
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    return fig

def plot_batch_predictions(predictions, class_names, true_labels=None):
    """Plot distribution of predictions for batch"""
    class_counts = {}
    for class_id in range(len(class_names)):
        class_counts[class_names[class_id]] = 0
    
    for pred in predictions:
        class_counts[pred['class_name']] += 1
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(class_counts.keys()),
            y=list(class_counts.values()),
            marker_color='lightcoral',
            text=list(class_counts.values()),
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Distribution of Predicted Classes',
        xaxis_title='Arrhythmia Classes',
        yaxis_title='Count',
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    return fig

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[class_names[i] for i in range(len(class_names))],
        y=[class_names[i] for i in range(len(class_names))],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=500,
        template='plotly_white'
    )
    
    return fig

# ================================
# Main Streamlit App
# ================================

def main():
    """Main Streamlit application"""
    setup_page()
    
    # Header
    st.markdown('<h1 class="main-header">🫀 ECG Arrhythmia Classification</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-powered ECG analysis using ResNet-50 Deep Learning</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Class names
    class_names = {
        0: 'N (Normal)', 
        1: 'S (Supraventricular)', 
        2: 'V (Ventricular)', 
        3: 'F (Fusion)', 
        4: 'Q (Unknown)'
    }
    
    # Model loading section
    st.sidebar.subheader("🤖 Model Configuration")
    
    # Fixed model path
    fixed_model_path = get_fixed_model_path()
    
    st.sidebar.info(f"**Using Trained ResNet-50 Model**")
    st.sidebar.code(fixed_model_path, language=None)
    
    model_option = st.sidebar.radio(
        "Model source:",
        ["Use trained ResNet-50", "Use default untrained model", "Upload custom model"]
    )
    
    model = None
    
    if model_option == "Use trained ResNet-50":
        if os.path.exists(fixed_model_path):
            model = load_saved_model(fixed_model_path)
            if model:
                st.sidebar.success(f"✅ Trained ResNet-50 loaded successfully!")
                
                # Show model info
                model_info = get_fixed_model_info()
                st.sidebar.markdown("**Model Information:**")
                st.sidebar.info(f"📋 **Model:** {model_info.get('model_name', 'ResNet-50')}")
                if 'test_accuracy' in model_info and model_info['test_accuracy'] != 'Trained Model':
                    st.sidebar.info(f"🎯 **Accuracy:** {model_info['test_accuracy']:.4f}")
                else:
                    st.sidebar.info(f"🎯 **Status:** Trained Model")
                if 'total_parameters' in model_info and model_info['total_parameters'] != 'Unknown':
                    st.sidebar.info(f"⚙️ **Parameters:** {model_info['total_parameters']:,}")
                st.sidebar.info(f"📅 **Timestamp:** {model_info.get('timestamp', 'Unknown')}")
        else:
            st.sidebar.error("❌ Trained model not found at specified path!")
            st.sidebar.warning("Check if the model file exists at:")
            st.sidebar.code(fixed_model_path)
            model = create_default_model()
            st.sidebar.warning("⚠️ Using default untrained model as fallback")
    
    elif model_option == "Use default untrained model":
        model = create_default_model()
        st.sidebar.warning("⚠️ Using default untrained ResNet-50 model")
        st.sidebar.info("This model has not been trained and will produce random predictions")
    
    else:  # Upload custom model
        uploaded_model = st.sidebar.file_uploader(
            "Upload model file (.pth)",
            type=['pth'],
            help="Upload a trained ResNet-50 model file"
        )
        
        if uploaded_model:
            try:
                model = torch.load(uploaded_model, map_location='cpu')
                model.eval()
                st.sidebar.success("✅ Custom model loaded successfully")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading custom model: {e}")
                model = create_default_model()
                st.sidebar.warning("⚠️ Using default model as fallback")
        else:
            st.sidebar.info("Please upload a .pth model file")
            model = create_default_model()
            st.sidebar.warning("⚠️ Using default model until custom model is uploaded")
    
    # Main content area
    if model is None:
        st.error("❌ Failed to load any model. Please check your model files.")
        return
    
    # File upload section
    st.markdown('<h2 class="sub-header">📁 Upload ECG Data</h2>', unsafe_allow_html=True)
    
    # Data format selection
    data_format = st.radio(
        "Select your data format:",
        ["Features only (187 columns)", "Features with labels (188 columns)"],
        help="Choose whether your CSV contains only ECG features or includes true labels for validation"
    )
    
    has_labels = (data_format == "Features with labels (188 columns)")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with ECG data",
        type=['csv'],
        help=f"Upload a CSV file where each row represents an ECG signal. Expected format: {'187 features + 1 label column' if has_labels else '187 feature columns'}"
    )
    
    if uploaded_file is not None:
        try:
            # Load CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ CSV loaded successfully! Shape: {df.shape}")
            
            # Validate data
            errors, warnings = validate_csv_data(df, has_labels)
            
            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
                return
            
            if warnings:
                for warning in warnings:
                    st.warning(f"⚠️ {warning}")
            
            # Show data preview
            with st.expander("📊 Data Preview", expanded=False):
                st.write("**First 5 rows:**")
                st.dataframe(df.head())
                
                st.write("**Data Statistics:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Missing Values", df.isnull().sum().sum())
                with col4:
                    st.metric("Has Labels", "Yes" if has_labels else "No")
            
            # Preprocess data
            with st.spinner("🔄 Preprocessing data..."):
                if has_labels:
                    processed_data, true_labels = preprocess_csv_data(df, has_labels=True)
                    st.success(f"✅ Data preprocessed! Ready for prediction with {processed_data.shape[0]} samples and labels")
                else:
                    processed_data, true_labels = preprocess_csv_data(df, has_labels=False)
                    st.success(f"✅ Data preprocessed! Ready for prediction with {processed_data.shape[0]} samples")
            
            # Prediction section
            st.markdown('<h2 class="sub-header">🔮 ECG Classification Results</h2>', unsafe_allow_html=True)
            
            if st.button("🚀 Run Predictions", type="primary"):
                with st.spinner("🤖 Making predictions..."):
                    predictions = predict_ecg_batch(model, processed_data, class_names)
                
                # Calculate metrics if labels are available
                if has_labels and true_labels is not None:
                    accuracy, class_report, cm = calculate_metrics(predictions, true_labels, class_names)
                
                # Results summary
                if has_labels:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Samples", len(predictions))
                    with col2:
                        st.metric("Accuracy", f"{accuracy:.3f}")
                    with col3:
                        correct_preds = sum(1 for i, p in enumerate(predictions) if p['class_id'] == true_labels[i])
                        st.metric("Correct", f"{correct_preds}/{len(predictions)}")
                    with col4:
                        avg_confidence = np.mean([p['confidence'] for p in predictions])
                        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                else:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Samples", len(predictions))
                    with col2:
                        avg_confidence = np.mean([p['confidence'] for p in predictions])
                        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                    with col3:
                        normal_count = sum(1 for p in predictions if p['class_name'] == 'N (Normal)')
                        st.metric("Normal Beats", f"{normal_count}/{len(predictions)}")
                
                # Visualization tabs
                if has_labels:
                    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                        "📈 Sample Signals", "📊 Prediction Distribution", 
                        "🎯 Individual Results", "📈 Model Performance", 
                        "🔥 Confusion Matrix", "📋 Detailed Report"
                    ])
                else:
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📈 Sample Signals", "📊 Prediction Distribution", 
                        "🎯 Individual Results", "📋 Detailed Report"
                    ])
                
                with tab1:
                    st.subheader("Sample ECG Signals")
                    
                    # Show configurable number of samples
                    max_samples = min(10, len(processed_data))
                    num_samples = st.slider("Number of samples to display:", 1, max_samples, min(5, max_samples))
                    
                    for i in range(num_samples):
                        if has_labels:
                            true_label_name = class_names[true_labels[i]]
                            pred_label_name = predictions[i]['class_name']
                            fig = plot_ecg_signal(
                                processed_data[i], 
                                f"ECG Sample {i+1}", 
                                i, 
                                true_label_name, 
                                pred_label_name
                            )
                        else:
                            fig = plot_ecg_signal(processed_data[i], f"ECG Sample {i+1}", i)
                        
                        st.plotly_chart(fig, use_container_width=True, key=f"ecg_plot_{i}")
                        
                        # Show prediction for this sample with updated format
                        pred = predictions[i]
                        if has_labels:
                            is_correct = pred['class_id'] == true_labels[i]
                            container_class = "correct" if is_correct else "incorrect"
                            accuracy_icon = "✅" if is_correct else "❌"
                            actual_label_text = f"<br><strong>Actual:</strong> {class_names[true_labels[i]]}"
                        else:
                            container_class = ""
                            accuracy_icon = ""
                            actual_label_text = ""
                        
                        confidence_color = "🟢" if pred['confidence'] > 0.8 else "🟡" if pred['confidence'] > 0.6 else "🔴"
                        st.markdown(f"""
                        <div class="prediction-container {container_class}">
                        <div class="prediction-text">
                        <strong>Predicted:</strong> {pred['class_name']} {confidence_color} {accuracy_icon}<br>
                        <strong>Confidence:</strong> {pred['confidence']:.3f}
                        {actual_label_text}
                        </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with tab2:
                    st.subheader("Prediction Distribution")
                    
                    fig_dist = plot_batch_predictions(predictions, class_names, true_labels)
                    st.plotly_chart(fig_dist, use_container_width=True, key="batch_predictions_chart")
                    
                    # Class statistics
                    st.subheader("Class Statistics")
                    class_stats = {}
                    for i, class_name in class_names.items():
                        count = sum(1 for p in predictions if p['class_name'] == class_name)
                        percentage = (count / len(predictions)) * 100
                        avg_conf = np.mean([p['confidence'] for p in predictions if p['class_name'] == class_name]) if count > 0 else 0
                        
                        # Add accuracy per class if labels available
                        if has_labels:
                            true_count = sum(1 for label in true_labels if label == i)
                            correct_count = sum(1 for j, p in enumerate(predictions) 
                                              if p['class_name'] == class_name and true_labels[j] == i)
                            class_accuracy = (correct_count / true_count) if true_count > 0 else 0
                            class_stats[class_name] = {
                                'predicted_count': count,
                                'true_count': true_count,
                                'correct_count': correct_count,
                                'percentage': percentage,
                                'avg_confidence': avg_conf,
                                'class_accuracy': class_accuracy
                            }
                        else:
                            class_stats[class_name] = {
                                'count': count,
                                'percentage': percentage,
                                'avg_confidence': avg_conf
                            }
                    
                    if has_labels:
                        stats_df = pd.DataFrame(class_stats).T
                        stats_df.columns = ['Predicted Count', 'True Count', 'Correct Count', 'Percentage (%)', 'Avg Confidence', 'Class Accuracy']
                        st.dataframe(stats_df.round(3))
                    else:
                        stats_df = pd.DataFrame(class_stats).T
                        stats_df.columns = ['Count', 'Percentage (%)', 'Avg Confidence']
                        st.dataframe(stats_df.round(3))
                
                with tab3:
                    st.subheader("Individual Prediction Results")
                    
                    # Pagination for large datasets
                    page_size = 10
                    total_pages = (len(predictions) + page_size - 1) // page_size
                    
                    if total_pages > 1:
                        page = st.selectbox("Select page", range(1, total_pages + 1)) - 1
                        start_idx = page * page_size
                        end_idx = min(start_idx + page_size, len(predictions))
                    else:
                        start_idx = 0
                        end_idx = len(predictions)
                    
                    for i in range(start_idx, end_idx):
                        pred = predictions[i]
                        
                        if has_labels:
                            is_correct = pred['class_id'] == true_labels[i]
                            status_icon = "✅" if is_correct else "❌"
                            actual_label_info = f" | Actual: {class_names[true_labels[i]]}"
                        else:
                            status_icon = ""
                            actual_label_info = ""
                        
                        with st.expander(f"Sample {i+1}: Predicted: {pred['class_name']} (Conf: {pred['confidence']:.3f}){actual_label_info} {status_icon}"):
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                if has_labels:
                                    true_label_name = class_names[true_labels[i]]
                                    pred_label_name = pred['class_name']
                                    fig_signal = plot_ecg_signal(
                                        processed_data[i], 
                                        f"ECG Sample {i+1}", 
                                        i, 
                                        true_label_name, 
                                        pred_label_name
                                    )
                                else:
                                    fig_signal = plot_ecg_signal(processed_data[i], f"ECG Sample {i+1}", i)
                                st.plotly_chart(fig_signal, use_container_width=True, key=f"individual_ecg_{i}")
                            
                            with col2:
                                true_class_id = true_labels[i] if has_labels else None
                                fig_probs = plot_prediction_probabilities(
                                    pred['probabilities'], 
                                    class_names, 
                                    i, 
                                    true_class_id
                                )
                                st.plotly_chart(fig_probs, use_container_width=True, key=f"individual_probs_{i}")
                
                if has_labels:
                    with tab4:
                        st.subheader("Model Performance Metrics")
                        
                        # Overall metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Overall Accuracy", f"{accuracy:.4f}")
                            st.metric("Macro Avg F1-Score", f"{class_report['macro avg']['f1-score']:.4f}")
                        with col2:
                            st.metric("Weighted Avg Precision", f"{class_report['weighted avg']['precision']:.4f}")
                            st.metric("Weighted Avg Recall", f"{class_report['weighted avg']['recall']:.4f}")
                        
                        # Per-class metrics
                        st.subheader("Per-Class Performance")
                        metrics_data = []
                        for i, class_name in class_names.items():
                            if str(i) in class_report:
                                metrics_data.append({
                                    'Class': class_name,
                                    'Precision': class_report[str(i)]['precision'],
                                    'Recall': class_report[str(i)]['recall'],
                                    'F1-Score': class_report[str(i)]['f1-score'],
                                    'Support': class_report[str(i)]['support']
                                })
                        
                        metrics_df = pd.DataFrame(metrics_data)
                        st.dataframe(metrics_df.round(4))
                    
                    with tab5:
                        st.subheader("Confusion Matrix")
                        
                        fig_cm = plot_confusion_matrix(cm, class_names)
                        st.plotly_chart(fig_cm, use_container_width=True, key="confusion_matrix_chart")
                        
                        # Additional confusion matrix insights
                        st.subheader("Confusion Matrix Analysis")
                        
                        # Most confused pairs
                        confused_pairs = []
                        for i in range(len(class_names)):
                            for j in range(len(class_names)):
                                if i != j and cm[i][j] > 0:
                                    confused_pairs.append({
                                        'True Class': class_names[i],
                                        'Predicted Class': class_names[j],
                                        'Count': cm[i][j],
                                        'Error Rate': cm[i][j] / cm[i].sum()
                                    })
                        
                        if confused_pairs:
                            confused_df = pd.DataFrame(confused_pairs)
                            confused_df = confused_df.sort_values('Count', ascending=False)
                            st.write("**Most Common Misclassifications:**")
                            st.dataframe(confused_df.head(10).round(4))
                
                # Determine which tab is the detailed report tab
                report_tab = tab6 if has_labels else tab4
                
                with report_tab:
                    st.subheader("Detailed Prediction Report")
                    
                    # Create detailed DataFrame
                    results_data = []
                    for i, pred in enumerate(predictions):
                        result_row = {
                            'Sample_ID': i + 1,
                            'Predicted_Class': pred['class_name'],
                            'Predicted_Class_ID': pred['class_id'],
                            'Confidence': round(pred['confidence'], 4),
                            'Normal_Prob': round(pred['probabilities'][0], 4),
                            'Supraventricular_Prob': round(pred['probabilities'][1], 4),
                            'Ventricular_Prob': round(pred['probabilities'][2], 4),
                            'Fusion_Prob': round(pred['probabilities'][3], 4),
                            'Unknown_Prob': round(pred['probabilities'][4], 4)
                        }
                        
                        if has_labels:
                            result_row['Actual_Class'] = class_names[true_labels[i]]
                            result_row['Actual_Class_ID'] = true_labels[i]
                            result_row['Correct'] = pred['class_id'] == true_labels[i]
                        
                        results_data.append(result_row)
                    
                    results_df = pd.DataFrame(results_data)
                    
                    # Reorder columns to show Actual right after Predicted when available
                    if has_labels:
                        column_order = ['Sample_ID', 'Predicted_Class', 'Actual_Class', 'Confidence', 'Correct', 
                                      'Predicted_Class_ID', 'Actual_Class_ID', 'Normal_Prob', 'Supraventricular_Prob', 
                                      'Ventricular_Prob', 'Fusion_Prob', 'Unknown_Prob']
                        results_df = results_df[column_order]
                    
                    st.dataframe(results_df)
                    
                    # Summary statistics
                    if has_labels:
                        st.subheader("Summary Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Samples", len(results_df))
                        with col2:
                            correct_count = results_df['Correct'].sum()
                            st.metric("Correct Predictions", f"{correct_count}/{len(results_df)}")
                        with col3:
                            st.metric("Overall Accuracy", f"{correct_count/len(results_df):.4f}")
                        with col4:
                            high_conf_correct = len(results_df[(results_df['Confidence'] > 0.8) & (results_df['Correct'] == True)])
                            high_conf_total = len(results_df[results_df['Confidence'] > 0.8])
                            if high_conf_total > 0:
                                st.metric("High Conf. Accuracy", f"{high_conf_correct/high_conf_total:.4f}")
                            else:
                                st.metric("High Conf. Accuracy", "N/A")
                    
                    # Download button
                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    filename_suffix = "_with_validation" if has_labels else "_predictions_only"
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_data,
                        file_name=f"ecg_predictions{filename_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            st.info("Please ensure your CSV file contains numeric ECG data with each row representing one ECG signal.")
    
    else:
        # Instructions when no file is uploaded
        st.markdown('<h2 class="sub-header">📝 Instructions</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### How to use this ECG Classification App:
        
        1. **Your trained ResNet-50 model is ready!**
           - Located at: `/home/irman/dskc_cnn/saved_models/ResNet-50_20250623_103251/`
           - The app automatically loads your trained model
        
        2. **Choose your data format:**
           - **Features only**: 187 columns (ECG time points only)
           - **Features with labels**: 188 columns (187 ECG features + 1 label column for validation)
        
        3. **Upload your ECG data** as a CSV file
           - Each row should represent one ECG heartbeat signal
           - Each signal should have 187 time points (columns)
           - Data should be numeric values
           - If using labels, last column should contain class IDs (0-4)
        
        4. **Expected CSV formats:**
           
           **Features only (187 columns):**
           ```
           0.123, 0.234, 0.345, ..., (187 values total)
           0.456, 0.567, 0.678, ..., (187 values total)
           ...
           ```
           
           **Features with labels (188 columns):**
           ```
           0.123, 0.234, 0.345, ..., (187 features), 0
           0.456, 0.567, 0.678, ..., (187 features), 2
           ...
           ```
        
        5. **The app will:**
           - Load your trained ResNet-50 model automatically
           - Validate your data format
           - Preprocess the signals automatically
           - Run predictions with your trained model
           - Display results with comprehensive visualizations
           - Show accuracy metrics if labels are provided
           - **Show Predicted, Actual, and Confidence for each sample**
        
        6. **Classification Classes (Label IDs):**
           - **0 - N (Normal)**: Normal heartbeats
           - **1 - S (Supraventricular)**: Atrial premature beats
           - **2 - V (Ventricular)**: Premature ventricular contractions
           - **3 - F (Fusion)**: Fusion of ventricular and normal beats
           - **4 - Q (Unknown)**: Paced beats or unclassifiable beats
        
        ### 🎯 Model Information
        - **Architecture**: ResNet-50 adapted for 1D ECG signals
        - **Training Dataset**: MIT-BIH Arrhythmia Database
        - **Input**: 187-point ECG signals
        - **Output**: 5-class arrhythmia classification
        
        ### 📊 Sample Data Format
        Generate sample data for testing:
        """)
        
        # Sample data generation section
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔬 Generate Sample ECG Data (Features Only)"):
                # Generate sample ECG data without labels
                np.random.seed(42)
                sample_data = []
                
                num_samples = st.selectbox("Number of samples:", [5, 10, 20, 50], index=1, key="samples_no_labels")
                
                for _ in range(num_samples):
                    # Create a simple ECG-like signal
                    t = np.linspace(0, 1, 187)
                    signal = np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(187)
                    signal = (signal - signal.min()) / (signal.max() - signal.min())  # Normalize
                    sample_data.append(signal)
                
                sample_df = pd.DataFrame(sample_data)
                
                st.success(f"✅ Sample ECG data generated! ({num_samples} samples)")
                st.info("This synthetic data is for testing purposes. Use real ECG data for actual diagnosis.")
                st.dataframe(sample_df.head())
                
                # Show a sample plot
                fig_sample = plot_ecg_signal(sample_data[0], "Sample ECG Signal", 0)
                st.plotly_chart(fig_sample, use_container_width=True, key="sample_ecg_plot_no_labels")
                
                # Download sample data
                csv_buffer = io.StringIO()
                sample_df.to_csv(csv_buffer, index=False)
                sample_csv = csv_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Sample Data (Features Only)",
                    data=sample_csv,
                    file_name="sample_ecg_features_only.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("🔬 Generate Sample ECG Data (With Labels)"):
                # Generate sample ECG data with labels
                np.random.seed(42)
                sample_data = []
                sample_labels = []
                
                num_samples = st.selectbox("Number of samples:", [5, 10, 20, 50], index=1, key="samples_with_labels")
                
                for i in range(num_samples):
                    # Create different ECG-like signals for different classes
                    label = i % 5  # Cycle through classes 0-4
                    t = np.linspace(0, 1, 187)
                    
                    if label == 0:  # Normal
                        signal = np.sin(2 * np.pi * 2 * t) + 0.3 * np.sin(2 * np.pi * 8 * t)
                    elif label == 1:  # Supraventricular
                        signal = np.sin(2 * np.pi * 3 * t) + 0.4 * np.sin(2 * np.pi * 12 * t)
                    elif label == 2:  # Ventricular
                        signal = 0.8 * np.sin(2 * np.pi * 1.5 * t) + 0.6 * np.sin(2 * np.pi * 6 * t)
                    elif label == 3:  # Fusion
                        signal = 0.6 * (np.sin(2 * np.pi * 2 * t) + np.sin(2 * np.pi * 1.5 * t))
                    else:  # Unknown
                        signal = 0.5 * np.sin(2 * np.pi * 4 * t) + 0.3 * np.random.randn(187)
                    
                    # Add noise and normalize
                    signal += 0.1 * np.random.randn(187)
                    signal = (signal - signal.min()) / (signal.max() - signal.min())
                    
                    sample_data.append(signal)
                    sample_labels.append(label)
                
                # Create DataFrame with features and labels
                sample_df = pd.DataFrame(sample_data)
                sample_df['Label'] = sample_labels
                
                st.success(f"✅ Sample ECG data with labels generated! ({num_samples} samples)")
                st.info("This synthetic data includes labels for validation testing.")
                
                # Show label distribution
                label_counts = pd.Series(sample_labels).value_counts().sort_index()
                st.write("**Label distribution:**")
                for label_id, count in label_counts.items():
                    st.write(f"- {class_names[label_id]}: {count} samples")
                
                st.dataframe(sample_df.head())
                
                # Show sample plots for different classes
                unique_labels = sorted(set(sample_labels))
                for label in unique_labels[:3]:  # Show first 3 classes
                    idx = sample_labels.index(label)
                    fig_sample = plot_ecg_signal(
                        sample_data[idx], 
                        f"Sample {class_names[label]} ECG Signal", 
                        idx
                    )
                    st.plotly_chart(fig_sample, use_container_width=True, key=f"sample_ecg_plot_label_{label}")
                
                # Download sample data
                csv_buffer = io.StringIO()
                sample_df.to_csv(csv_buffer, index=False)
                sample_csv = csv_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Sample Data (With Labels)",
                    data=sample_csv,
                    file_name="sample_ecg_with_labels.csv",
                    mime="text/csv"
                )

# ================================
# Run the app
# ================================

if __name__ == "__main__":
    main()