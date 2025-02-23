import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency

sns.set(style='dark')

# Function to preprocess data
def preprocess_data(df):
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'], errors='coerce')
    df['month'] = df['order_purchase_timestamp'].dt.to_period('M').astype(str)
    df['total_spent'] = df['price'] + df['freight_value']
    return df

# Function to plot monthly sales
def plot_monthly_sales(df):
    monthly_sales = df.groupby('month').agg(total_sales=('price', 'sum')).reset_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='month', y='total_sales', data=monthly_sales, marker='o', color='Blue', linewidth=2.5, markersize=8)
    plt.title('Total Penjualan Per Bulan', fontsize=16)
    plt.xlabel('Bulan', fontsize=12)
    plt.ylabel('Total Penjualan (R$)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return plt

# Function to plot rating distribution
def plot_rating_distribution(df):
    rating_counts = df['review_score'].value_counts().sort_index()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=rating_counts.index, y=rating_counts.values, palette='coolwarm')
    plt.title('Distribusi Rating Ulasan Pelanggan', fontsize=16)
    plt.xlabel('Rating', fontsize=12)
    plt.ylabel('Jumlah Ulasan', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return plt

# Function to plot payment methods
def plot_payment_methods(df):
    payment_counts = df['payment_type'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set3"))
    plt.title('Metode Pembayaran yang Paling Sering Digunakan', fontsize=16)
    plt.tight_layout()
    return plt

# Function to get top customers
def get_top_customers(df, n=10):
    top_customers = df.groupby('customer_id').agg(total_spent=('price', 'sum')).reset_index()
    top_customers = top_customers.sort_values(by='total_spent', ascending=False).head(n)
    return top_customers

# Function to plot top customers
def plot_top_customers(df, n=10):
    top_customers = get_top_customers(df, n)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='total_spent', y='customer_id', data=top_customers, palette='rocket', edgecolor='black', alpha=0.8, orient='h')
    plt.title('Top 10 Pelanggan Terbaik', fontsize=16)
    plt.xlabel('Total Pengeluaran (R$)', fontsize=12)
    plt.ylabel('ID Pelanggan', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return plt

# Function to get top positive reviewers
def get_top_positive_reviewers(df, n=10):
    positive_reviews = df[df['review_score'] >= 4]
    positive_reviews_by_customer = positive_reviews.groupby('customer_id').size().reset_index(name='positive_reviews_count')
    top_positive_customers = positive_reviews_by_customer.sort_values(by='positive_reviews_count', ascending=False).head(n)
    return top_positive_customers

# Function to plot top positive reviewers
def plot_top_positive_reviewers(df, n=10):
    top_positive_customers = get_top_positive_reviewers(df, n)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='positive_reviews_count', y='customer_id', data=top_positive_customers, palette='rocket', edgecolor='black', alpha=0.8, orient='h')
    plt.title('Karakteristik Pelanggan dengan Ulasan Positif', fontsize=16)
    plt.xlabel('Jumlah Ulasan Positif', fontsize=12)
    plt.ylabel('ID Pelanggan', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return plt

# Function to plot correlation matrix
def plot_correlation_matrix(df):
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    # Drop ID-like columns
    columns_to_exclude = ['customer_id', 'order_id', 'product_id']
    numeric_df = numeric_df.drop(columns=columns_to_exclude, errors='ignore')
    # Calculate correlation matrix
    correlation_matrix = numeric_df.corr()
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix, 
        annot=True, 
        cmap='viridis', 
        fmt='.2f', 
        linewidths=0.5, 
        linecolor='gray', 
        cbar_kws={"shrink": .8},
        annot_kws={"fontsize": 10, "color": "black"}
    )
    plt.title('Correlation Matrix (Excluding IDs)', fontsize=18, fontweight='bold', pad=20)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    return plt

# Function to perform RFM Analysis
def perform_rfm_analysis(df):
    # Recency
    latest_date = df['order_purchase_timestamp'].max()
    recency_df = df.groupby('customer_id').agg({
        'order_purchase_timestamp': lambda x: (latest_date - x.max()).days
    }).reset_index()
    recency_df.rename(columns={'order_purchase_timestamp': 'Recency'}, inplace=True)
    
    # Frequency
    frequency_df = df.groupby('customer_id').agg({
        'order_id': 'nunique'  # Jumlah pesanan unik
    }).reset_index()
    frequency_df.rename(columns={'order_id': 'Frequency'}, inplace=True)
    
    # Monetary
    monetary_df = df.groupby('customer_id').agg({
        'total_spent': 'sum'
    }).reset_index()
    monetary_df.rename(columns={'total_spent': 'Monetary'}, inplace=True)
    
    # Merge RFM dataframes
    rfm_df = pd.merge(recency_df, frequency_df, on='customer_id')
    rfm_df = pd.merge(rfm_df, monetary_df, on='customer_id')
    
    # Score RFM
    rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
    
    # Frequency with custom bins
    bins = [0, 1, 2, 3, 4, float('inf')]
    labels = [1, 2, 3, 4, 5]
    rfm_df['F_Score'] = pd.cut(rfm_df['Frequency'], bins=bins, labels=labels)
    
    rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    
    # Total RFM Score
    rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(int) + rfm_df['F_Score'].astype(int) + rfm_df['M_Score'].astype(int)
    
    # Segment Customers
    def segment_customer(row):
        if row['RFM_Score'] >= 12:
            return 'Champions'
        elif 9 <= row['RFM_Score'] < 12:
            return 'Loyal Customers'
        elif 6 <= row['RFM_Score'] < 9:
            return 'Potential Loyalists'
        elif 3 <= row['RFM_Score'] < 6:
            return 'At Risk'
        else:
            return 'Lost Customers'
    
    rfm_df['Segment'] = rfm_df.apply(segment_customer, axis=1)
    
    return rfm_df

# Function to plot customer segments
def plot_customer_segments(rfm_df):
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=rfm_df, x='Segment', palette='coolwarm', order=rfm_df['Segment'].value_counts().index)
    
    # Annotate bars
    for bar in ax.patches:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 50,
            f'{int(height)}',
            ha='center',
            fontsize=10,
            color='black'
        )
    
    plt.title('Distribusi Segmen Pelanggan Berdasarkan RFM', fontsize=16)
    plt.xlabel('Segment', fontsize=12)
    plt.ylabel('Jumlah Pelanggan', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return plt

# Load and preprocess data
file_path = 'main_data.csv'
df = pd.read_csv(file_path)
df = preprocess_data(df)

# Menentukan rentang tanggal minimum dan maksimum
min_date = df['order_purchase_timestamp'].min().date()
max_date = df['order_purchase_timestamp'].max().date()

# Sidebar untuk filter tanggal
st.sidebar.header("Filter Tanggal")
start_date, end_date = st.sidebar.date_input("Pilih Rentang Tanggal", [min_date, max_date])

# Mengfilter data berdasarkan rentang tanggal
filtered_df = df[(df['order_purchase_timestamp'].dt.date >= start_date) & (df['order_purchase_timestamp'].dt.date <= end_date)]

# Title of the dashboard
st.title("Olist Marketplace Analysis Dashboard")

# Sidebar for navigation
st.sidebar.header("Navigasi")
option = st.sidebar.selectbox(
    "Pilih Visualisasi Data",
    [
        "Penjualan Bulanan",
        "Distribusi Rating",
        "Metode Pembayaran",
        "Pelanggan Terbaik",
        "Ulasan Positif",
        "Korelasi",
        "Segmentasi RFM"
    ]
)

# Display visualizations based on selection
if option == "Penjualan Bulanan":
    st.header("Total Penjualan Per Bulan")
    fig = plot_monthly_sales(filtered_df)
    st.pyplot(fig)

elif option == "Distribusi Rating":
    st.header("Distribusi Rating Ulasan Pelanggan")
    fig = plot_rating_distribution(filtered_df)
    st.pyplot(fig)

elif option == "Metode Pembayaran":
    st.header("Metode Pembayaran yang Paling Sering Digunakan")
    fig = plot_payment_methods(filtered_df)
    st.pyplot(fig)

elif option == "Pelanggan Terbaik":
    st.header("Top 10 Pelanggan Terbaik")
    fig = plot_top_customers(filtered_df)
    st.pyplot(fig)

elif option == "Ulasan Positif":
    st.header("Karakteristik Pelanggan dengan Ulasan Positif")
    fig = plot_top_positive_reviewers(filtered_df)
    st.pyplot(fig)

elif option == "Korelasi":
    st.header("Matriks Korelasi")
    fig = plot_correlation_matrix(filtered_df)
    st.pyplot(fig)

elif option == "Segmentasi RFM":
    st.header("Segmentasi Pelanggan Berdasarkan RFM")
    rfm_df = perform_rfm_analysis(filtered_df)
    fig = plot_customer_segments(rfm_df)
    st.pyplot(fig)
    
    # Show RFM DataFrame
    st.subheader("RFM Segmentation Details")
    st.dataframe(rfm_df.head(10))
    
st.caption("Copyright (c) Marsello Ormanda 2025")