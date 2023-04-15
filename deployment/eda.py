import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
from PIL import Image


# Untuk melebarkan streamlit, harus diletakkan setelh import
# Ketika dieksekusi akan mempengaruhi main dan prediction
# Tidak perlu dijalankan dalam fungsi
st.set_page_config(
    page_title='Product Quantities Analysis and Forecasting',
    layout='wide',
    initial_sidebar_state='expanded'
)


# bagian bawah ini tidak bisa dijalankan jika tidak dieksekusi
def run():
    #Membuat Title
    st.title('Product Quantities Analysis and Forecasting Using Linear Regression')

    # Membuat Sub Header
    st.subheader('EDA for The Sales of The Products')

    # Menambahkan Gambar
    image=Image.open('product.jpg')
    st.markdown(
    """
    <style>
    img {
        cursor: pointer;
        transition: all .2s ease-in-out;
    }
    img:hover {
        transform: scale(1.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    st.image(image, caption='Product Analysis and Forecasting')

    # Menambah Deskripsi
    st.write('Made by *Happy Trianna*')
    
    # Membuat Garis Lurus
    st.markdown('---')

    # Magic Syntax
    '''
    On this page, the author will explore the product dataset of ParagonCorp company.
    '''

    # Show DataFrame
    st.write('#### Product Dataset')
    df = pd.read_csv('sample_dataset_timeseries_noarea.csv', encoding = "ISO-8859-1")
    st.dataframe(df.head(10))

    # Change the data type of the date into datetime
    df['week_start_date']=pd.to_datetime(df['week_start_date'])
    df['week_end_date']=pd.to_datetime(df['week_end_date'])

    # Print the range of data based on the first day of the week
    st.write("First data recorded on", df['week_start_date'].min().strftime('`%d %B %Y`'))
    st.write("Recent data recorded on", df['week_end_date'].max().strftime('`%d %B %Y`'))

    # Check the unique item of the products and the total sold unique products
    st.write("#### Preview of unique item sold :",df.product_item.unique())
    st.write("Total unique item sold =",df.product_item.nunique())

    # Get the week and year of the week_number column
    df['week'] = pd.to_numeric(df['week_number'].apply(lambda x: x[-2:]))
    df['year'] = pd.to_numeric(df['week_number'].apply(lambda x: x[:-3]))

    # Quantities of product sold per week
    st.write('#### Describe of Quantities of Sold Products Per Week')
    st.dataframe(df.groupby('week_start_date')['quantity'].sum().describe())
    st.write('It shows that the total of products sold per week are around 4.9M in 50% percentile and mean of the data, \
             while the maximum sold per week are 7.1M and minimum sold are 1.2M.')
    
    # Plot of number of sales and sum of products sold
    fig, ax = plt.subplots(nrows=2,figsize=(15,14))
    df.groupby('week_start_date')['quantity'].sum().plot(ax=ax[0])
    ax[0].set_ylabel('Quantity')
    ax[0].set_title("Products Sold by Quantity per Week", fontsize=16)
    sns.countplot(data=df, x="week", hue="year", ax=ax[1])
    ax[1].set_title("Number of Times Sales Per Week", fontsize=16)
    st.pyplot(fig)
    # Get the date of week 18 from year 2022
    st.write('Week 18 of 2022 = ',datetime.strptime('2022-18' + '-1', "%Y-%W-%w"))
    st.markdown('''
    - In 2022, `sales in week 18`, or the first week of May, appear to be `lower than in other weeks`, judging by the appearance of fewer products sold than in other weeks. `May 2nd, 2022 coincided with Eid al-Fitr`, so we assumed that `many employees were not working` during the collective leave schedule, resulting in less productivity than usual. The impact of total numbers of sales decreasing, `resulting in a drastic decrease in the number of products sold`, dropped to 1.2M.
    - In 2021, it appears that the total number of sales is much less than in other weeks, it can be assumed that `December 27` is the date `when the initial product sales are recorded`, so it is possible that the data at the beginning is incomplete or the sales promotion has just started.
    ''')

    # Check the maximum and minimum quantities of each sold item
    st.dataframe(df['product_item'].value_counts())
    st.markdown('''
    - Maximum items sold for each product are 67
    - Minimum items sold for each product is 1
    ''')


if __name__ == '__main__':
    run()