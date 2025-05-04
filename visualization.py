import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def plot_feature_importance(model, feature_names):
    """
    Create a bar chart of feature importance for a logistic regression model
    
    Parameters:
    ----------
    model : sklearn.linear_model.LogisticRegression
        Trained logistic regression model
    feature_names : array-like
        Names of the features
    
    Returns:
    -------
    plotly.graph_objects.Figure
        Feature importance plot
    """
    # Get coefficients from the model
    coefficients = model.coef_[0]
    
    # Create a DataFrame for plotting
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(coefficients)
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Create color scale based on coefficient sign
    colors = ['#ff4b4b' if c < 0 else '#00cc96' for c in coefficients]
    
    # Create the plot
    fig = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance for Financial Fitness',
        color_discrete_sequence=['#00cc96']
    )
    
    fig.update_layout(
        xaxis_title='Importance (Absolute Coefficient Value)',
        yaxis_title='Financial Factor',
        template='plotly_dark'
    )
    
    return fig

def plot_confusion_matrix(conf_matrix):
    """
    Create a heatmap visualization of a confusion matrix
    
    Parameters:
    ----------
    conf_matrix : numpy.ndarray
        Confusion matrix from model evaluation
    
    Returns:
    -------
    plotly.graph_objects.Figure
        Confusion matrix plot
    """
    # Labels for the confusion matrix
    labels = ['Unfit', 'Fit']
    
    # Create the heatmap
    fig = px.imshow(
        conf_matrix,
        x=labels,
        y=labels,
        color_continuous_scale=['#ff4b4b', '#00cc96'],
        labels=dict(x="Predicted", y="Actual", color="Count")
    )
    
    # Add text annotations
    annotations = []
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix[i])):
            annotations.append({
                'x': labels[j],
                'y': labels[i],
                'text': str(conf_matrix[i, j]),
                'showarrow': False,
                'font': {'color': 'white', 'size': 16}
            })
    
    fig.update_layout(
        title='Confusion Matrix',
        annotations=annotations,
        template='plotly_dark'
    )
    
    return fig

def plot_stock_data(stock_data):
    """
    Create a candlestick chart for stock price data
    
    Parameters:
    ----------
    stock_data : pandas.DataFrame
        Stock price data from Yahoo Finance
    
    Returns:
    -------
    plotly.graph_objects.Figure
        Stock price candlestick chart
    """
    # Create the candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        increasing_line_color='#00cc96',
        decreasing_line_color='#ff4b4b'
    )])
    
    # Add volume bars
    fig.add_trace(go.Bar(
        x=stock_data.index,
        y=stock_data['Volume'],
        marker_color='rgba(128, 128, 128, 0.3)',
        name='Volume',
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title='Stock Price Movement',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False
        )
    )
    
    return fig
