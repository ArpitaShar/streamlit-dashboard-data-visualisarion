# import libraries
import streamlit as st
# import os
# import glob
import pandas as pd
# import datetime
import numpy as np
import geopandas as gpd
import leafmap.foliumap as leafmap
import mapclassify
import matplotlib.pyplot as plt 
import plotly.graph_objects as go


st.set_page_config(layout='wide', initial_sidebar_state='expanded')

st.title("🛰️🌧️ Weather Dashboard 📊📈")

daily_rain_2023 = r"F:\Automation_Working\IMD_data\xls\daily_rainfall_2023.csv"
normal_rain = r"F:\Automation_Working\IMD_data\xls\normal_rainfall.csv"
district_shp_path = r"F:\Automation_Working\shapefiles\india_districts_gcs_v2.shp"

# Import daily rainfall data
@st.cache_data
def import_rain_data():
    data = pd.read_csv(daily_rain_2023)
    # Convert date formats from str to datetime object
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    data['month'] = data.Date.dt.month
    data['day'] = data.Date.dt.day
    data['Date'] = data['Date'].dt.date
    states_list = data['stname'].unique().tolist()
    states_list.sort()
    return data, states_list

# Import normal rainfall data
@st.cache_data
def import_normal_data():
    data = pd.read_csv(normal_rain)
    return data

# Function to merge normal rainfall with daily actual rainfall data
@st.cache_data
def mergeActualNormal(df1, df2):
    df = pd.merge(df1, df2, on=['FID', 'month', 'day'], how='inner')
    df = df.drop(columns=['month', 'day'])
    # df['Departure'] = (df['Actual Rainfall'] - df['Normal Rainfall']) / df['Normal Rainfall'] * 100
    return df

# Function to create weekly data from daily data
@st.cache_data
def createWeekly(df):
    df['Date'] = pd.to_datetime(df['Date'])
    resampled_df = df.groupby(['FID', 'stname', 'dtname', pd.Grouper(key='Date', freq='7D')]).sum(numeric_only=True).reset_index()
    dates_dict = dict(enumerate(resampled_df.Date.unique()))
    return resampled_df, dates_dict

# Function to create fortnightly data from daily data
@st.cache_data
def createFortnightly(df):
    df['Date'] = pd.to_datetime(df['Date'])
    fn_values = np.where(df['Date'].dt.day < 16, '1FN', '2FN')
    df['Fortnight'] = df['Date'].dt.strftime('%b ') + fn_values
    df['Fortnight'] = np.where((df['Date'].dt.month == 2) & (df['Date'].dt.day > 14), 'Feb 2FN', df['Fortnight'])
    resampled_df = df.groupby(['FID', 'stname', 'dtname', 'Fortnight']).sum(numeric_only=True).reset_index()
    return resampled_df

# Function to create monthly data from daily data
@st.cache_data
def createMonthly(df):
    df['Date'] = pd.to_datetime(df['Date'])
    resampled_df = df.groupby(['FID', 'stname', 'dtname', pd.Grouper(key='Date', freq='M')]).sum(numeric_only=True).reset_index()
    dates_dict = dict(enumerate(resampled_df.Date.unique()))
    return resampled_df, dates_dict

# Import shapefile as geopandas dataframe
@st.cache_data
def import_shp(shp_path):
    gdf = gpd.read_file(shp_path)
    return gdf

# Function to join and plot actual rainfall statistics to vector data (except fortnightly data) 
def add_actual_vector(shp_gdf, filtered_df, intervals):
    if len(filtered_df.index) > 0:
        filtered_df = filtered_df.drop(columns=['Date', 'stname', 'dtname'])
        vector_join_gdf = pd.merge(shp_gdf, filtered_df, on='FID', how='inner')
        vector_join_gdf = vector_join_gdf.drop(columns=['_Subdiv_no'])
        # st.write(vector_join_gdf)

        m = leafmap.Map(center=[20.5937, 78.9629], zoom=4)
        m.add_basemap(google_map="HYBRID", show=True)
        m.add_data(
            vector_join_gdf, column='Actual Rainfall', scheme='UserDefined', k=8, cmap='Blues', legend_title='Actual Rainfall (mm)', classification_kwds=dict(bins=intervals), layer_name='Actual Rainfall'
        )
        m.to_streamlit()

# Function to join and plot normal rainfall statistics to vector data (except fortnightly data) 
def add_normal_vector(shp_gdf, filtered_df, intervals):
    if len(filtered_df.index) > 0:
        filtered_df = filtered_df.drop(columns=['Date', 'stname', 'dtname'])
        vector_join_gdf = pd.merge(shp_gdf, filtered_df, on='FID', how='inner')
        vector_join_gdf = vector_join_gdf.drop(columns=['_Subdiv_no'])
        m = leafmap.Map(center=[20.5937, 78.9629], zoom=4)
        m.add_basemap(google_map="HYBRID", show=True)
        m.add_data(
            vector_join_gdf, column='Normal Rainfall', scheme='UserDefined', k =8, cmap='Blues', legend_title='Normal Rainfall (mm)', classification_kwds=dict(bins=intervals), layer_name='Normal Rainfall'
        )
        m.to_streamlit()

# Function to join and plot departure rainfall statistics to vector data (except fortnightly data) 
def add_departure_vector(shp_gdf, filtered_df, intervals):
    if len(filtered_df.index) > 0:
        filtered_df = filtered_df.drop(columns=['Date', 'stname', 'dtname'])
        vector_join_gdf = pd.merge(shp_gdf, filtered_df, on='FID', how='inner')
        vector_join_gdf = vector_join_gdf.drop(columns=['Subdiv_no', 'concat'])
        m = leafmap.Map(center=[20.5937, 78.9629], zoom=4)
        m.add_basemap(google_map="HYBRID", show=True)
        m.add_data(
            vector_join_gdf, column='Departure', scheme='UserDefined', k=7, cmap='Blues', legend_title='Departure (%)', classification_kwds=dict(bins=intervals), layer_name='% Departure'
        )
        m.to_streamlit()

# Function to join and plot fortnightly actual rainfall statistics to vector data
def add_fortnight_actual_vector(shp_gdf, filtered_df, intervals):
    if len(filtered_df.index) > 0:
        filtered_df = filtered_df.drop(columns=['Fortnight', 'stname', 'dtname'])
        vector_join_gdf = pd.merge(shp_gdf, filtered_df, on='FID', how='inner')
        vector_join_gdf = vector_join_gdf.drop(columns=['_Subdiv_no'])
        m = leafmap.Map(center=[20.5937, 78.9629], zoom=4)
        m.add_basemap(google_map="HYBRID", show=True)
        m.add_data(
            vector_join_gdf, column='Actual Rainfall', scheme='UserDefined', k=8, cmap='Blues', legend_title='Actual Rainfall (mm)', classification_kwds=dict(bins=intervals), layer_name='Actual Rainfall'
        )
        m.to_streamlit()

# Function to join and plot fortnightly normal rainfall statistics to vector data
def add_fortnight_normal_vector(shp_gdf, filtered_df, intervals):
    if len(filtered_df.index) > 0:
        filtered_df = filtered_df.drop(columns=['Fortnight', 'stname', 'dtname'])
        vector_join_gdf = pd.merge(shp_gdf, filtered_df, on='FID', how='inner')
        vector_join_gdf = vector_join_gdf.drop(columns=['_Subdiv_no'])
        m = leafmap.Map(center=[20.5937, 78.9629], zoom=4)
        m.add_basemap(google_map="HYBRID", show=True)
        m.add_data(
            vector_join_gdf, column='Normal Rainfall', scheme='UserDefined', k=8, cmap='Blues', legend_title='Normal Rainfall (mm)', classification_kwds=dict(bins=intervals), layer_name='Normal Rainfall'
        )
        m.to_streamlit()


# Function to plot data using plotly
def plot_lines(df):
    fig = go.Figure()
    x = df['dtname']
    y1 = df['Actual Rainfall']
    y2 = df['Normal Rainfall']

    fig.add_trace(go.Scatter(x=x, y=y1,
                    mode='lines+markers',
                    name='Actual Rainfall'))
    fig.add_trace(go.Scatter(x=x, y=y2,
                    mode='lines+markers',
                    name='Normal Rainfall'))
    fig.update_layout(showlegend=True,
        legend=dict(
            yanchor='top',
            y=.95,
            xanchor='left',
            x=0.01
        ),
        font_family="Arial"
    )
    fig.update_xaxes(tickangle=-60)
    
    st.plotly_chart(fig, theme=None, use_container_width=True)

# Function to plot data using plotly with normal as line and actual as bar
def plot_bars(df):
    x = df['dtname']
    y1 = df['Actual Rainfall']
    y2 = df['Normal Rainfall']

    fig = go.Figure(data=[
        go.Bar(name='Actual Rainfall', x=x, y=y1, marker_color='rgba(0, 161, 255, 0.8)',
               marker_line=dict(color='rgba(0, 161, 255, 1)', width=3)),
        go.Scatter(x=x, y=y2, mode='lines+markers', name='Normal Rainfall')
    ])

    # Change the bar mode
    fig.update_layout(barmode='group')

    # set the template to our custom_dark template
    # fig.layout.template = 'plotly_dark'
    fig.update_layout(showlegend=True,
                      legend=dict(
                          yanchor='top',
                          y=0.95,
                          xanchor='left',
                          x=0.01
                      ),
                      font_family="verdana",
                      height=700
                      )
    # fig.update_xaxes(tickangle=-20)
    st.plotly_chart(fig, use_container_width=True, theme=None)
    # print(df)

week_dict = {'Week 1 (01 Jan - 07 Jan)': 0,
             'Week 2 (08 Jan - 14 Jan)': 1,
             'Week 3 (15 Jan - 21 Jan)': 2,
             'Week 4 (22 Jan - 28 Jan)': 3,
             'Week 5 (29 Jan - 04 Feb)': 4,
             'Week 6 (05 Feb - 11 Feb)': 5,
             'Week 7 (12 Feb - 18 Feb)': 6,
             'Week 8 (19 Feb - 25 Feb)': 7,
             'Week 9 (27 Feb - 05 Mar)': 8,
             'Week 10 (06 Mar - 11 Mar)': 9,
             'Week 11 (12 Mar - 18 Mar)': 10,
             'Week 12 (19 Mar - 25 Mar)': 11,
             'Week 13 (26 Mar - 01 Apr)': 12,
             'Week 14 (02 Apr - 08 Apr)': 13,
             'Week 15 (09 Apr - 15 Apr)': 14,
             'Week 16 (16 Apr - 22 Apr)': 15,
             'Week 17 (23 Apr - 29 Apr)': 16,
             'Week 18 (30 Apr - 06 May)': 17,
             'Week 19 (07 May - 13 May)': 18,
             'Week 20 (14 May - 20 May)': 19,
             'Week 21 (21 May - 27 May)': 20,
             'Week 22 (28 May - 03 Jun)': 21,
             'Week 23 (04 Jun - 10 Jun)': 22,
             'Week 24 (11 Jun - 17 Jun)': 23,
             'Week 25 (18 Jun - 24 Jun)': 24,
             'Week 26 (25 Jun - 01 Jul)': 25,
             'Week 27 (02 Jul - 08 Jul)': 26,
             'Week 28 (09 Jul - 15 Jul)': 27,
             'Week 29 (16 Jul - 22 Jul)': 28,
             'Week 30 (23 Jul - 29 Jul)': 29,
             'Week 31 (30 Jul - 05 Aug)': 30,
             'Week 32 (06 Aug - 12 Aug)': 31,
             'Week 33 (13 Aug - 19 Aug)': 32,
             'Week 34 (20 Aug - 26 Aug)': 33,
             'Week 35 (27 Aug - 02 Sep)': 34,
             'Week 36 (03 Sep - 09 Sep)': 35,
             'Week 37 (10 Sep - 16 Sep)': 36,
             'Week 38 (17 Sep - 23 Sep)': 37,
             'Week 39 (24 Sep - 30 Sep)': 38,
             'Week 40 (01 Oct - 07 Oct)': 39,
             'Week 41 (08 Oct - 14 Oct)': 40,
             'Week 42 (15 Oct - 21 Oct)': 41,
             'Week 43 (22 Oct - 28 Oct)': 42,
             'Week 44 (29 Oct - 04 Nov)': 43,
             'Week 45 (05 Nov - 11 Nov)': 44,
             'Week 46 (12 Nov - 18 Nov)': 45,
             'Week 47 (19 Nov - 25 Nov)': 46,
             'Week 48 (26 Nov - 02 Dec)': 47,
             'Week 49 (03 Dec - 09 Dec)': 48,
             'Week 50 (10 Dec - 16 Dec)': 49,
             'Week 51 (17 Dec - 23 Dec)': 50,
             'Week 52 (24 Dec - 31 Dec)': 51,            

             }

month_dict = {'January': 0,
              'February': 1,
              'March': 2,
              'April': 3,
              'May': 4,
              'June': 5,
              'July': 6,
              'August': 7,
              'September': 8,
              'October': 9,
              'November': 10,
              'December': 11}


# Load and calculate rainfall data
rain_df = import_rain_data()[0]
normal_df = import_normal_data()
merged_df = mergeActualNormal(rain_df, normal_df)
weekly_df = createWeekly(merged_df)[0]
weekly_dates_dict = createWeekly(merged_df)[1]
fortnightly_df = createFortnightly(merged_df)
monthly_df = createMonthly(merged_df)[0]
monthly_dates_dict = createMonthly(merged_df)[1]

# Load shapefiles
district_gdf = import_shp(district_shp_path)

# States list
states_list = import_rain_data()[1]

# Add State selectbox to the sidebar:
state = st.sidebar.selectbox('Select State', states_list, key='state')

# Add time period selector to the sidebar
time_period = st.sidebar.radio('Select time period:', ['Daily', 'Weekly', 'Fortnightly', 'Monthly'], key='time_period')

# create classfiied intervals for different datasets
daily_intervals = [0, 10, 20, 50, 100, 150, 200, 500]
weekly_intervals = [0, 50, 100, 200, 300, 400, 500, 850]
fortnightly_intervals = [0, 100, 200, 400, 600, 800, 1000, 1500]
monthly_intervals = [0, 100, 300, 600, 800, 1000, 1500, 2500]

if st.session_state.time_period == 'Daily':
    st.subheader(':blue[Daily Rainfall Statistics]')
    date = st.sidebar.date_input('Select Date:', key='date') # Add Calendar to sidebar

    # Filter daily rainfall data of selected state
    daily_filtered = merged_df.loc[(merged_df['Date'] == st.session_state.date) & (merged_df['stname'] == st.session_state.state)]
    daily_filtered = daily_filtered.sort_values(by=['dtname'], ascending=True)
    # st.write(daily_filtered.columns)

    if len(daily_filtered.index) > 0:
        # Add columns to plot actual and normal rainfall
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Actual Rainfall Map")
            # actual_gdf = daily_filtered.copy()
            # Plot district shapefile with daily rainfall statistics
            add_actual_vector(district_gdf, daily_filtered, daily_intervals)
        with col2:
            st.subheader("Normal Rainfall Map")
            # normal_gdf = daily_filtered.copy()
            # Plot district shapefile with daily rainfall statistics
            add_normal_vector(district_gdf, daily_filtered, daily_intervals)
            # plot_lines(daily_filtered)
        
        # Plot bar plot of actual and normal rainfall
        plot_bars(daily_filtered)

        # Export daily data as CSV
        date_str = date.strftime("%Y-%m-%d")
        out_fn = daily_filtered['stname'].iloc[0] + '_' + date_str + '.csv' # Output filename
        dload_data = daily_filtered.to_csv(index=False).encode("utf-8")
        export_btn = st.sidebar.download_button('Export CSV', data=dload_data, file_name=out_fn, mime='text/csv', key='export_daily')
elif st.session_state.time_period == 'Weekly':   
    st.subheader(':blue[Weekly Rainfall Statistics]')
    week = st.sidebar.selectbox('Select Week:', week_dict) # Add Week widget to sidebar  
    week_no = week_dict.get(week) # Get week no. from week dictionary

    # Filter weekly rainfall data of selected state
    weekly_filtered = weekly_df.loc[(weekly_df['Date'] == weekly_dates_dict.get(week_no)) & (weekly_df['stname'] == st.session_state.state)]
    weekly_filtered = weekly_filtered.sort_values(by=['dtname'], ascending=True)

    if len(weekly_filtered.index) > 0:
        # Add columns to plot actual and normal rainfall
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Actual Rainfall Map")
            # Plot district shapefile with daily rainfall statistics
            add_actual_vector(district_gdf, weekly_filtered, weekly_intervals)
        with col2:
            st.subheader("Normal Rainfall Map")
            # Plot district shapefile with daily rainfall statistics
            add_normal_vector(district_gdf, weekly_filtered, weekly_intervals)
        plot_bars(weekly_filtered)

        # Export daily data as CSV
        weekly_filtered = weekly_filtered.drop(columns='Date')
        weekly_filtered['Week'] = week
        out_fn = weekly_filtered['stname'].iloc[0] + '_' + week + '.csv' # Output filename
        dload_data = weekly_filtered.to_csv(index=False).encode("utf-8")
        export_btn = st.sidebar.download_button('Export CSV', data=dload_data, file_name=out_fn, mime='text/csv', key='export_weekly')
elif st.session_state.time_period == 'Fortnightly':
    st.subheader(':blue[Fortnightly Rainfall Statistics]')
    month = st.sidebar.selectbox('Select Month:', ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']) # Add Month widget to sidebar
    fortnight = st.sidebar.selectbox('Select Fortnight:', ['1', '2']) # Add Fortnight widget to sidebar
    fortnight_id = month + ' ' + fortnight + 'FN' # Create fortnight ID to filter data

    # Filter fortnightly rainfall data of selected state
    fortnightly_filtered = fortnightly_df.loc[(fortnightly_df['Fortnight'] == fortnight_id) & (fortnightly_df['stname'] == st.session_state.state)]
    fortnightly_filtered = fortnightly_filtered.sort_values(by=['dtname'], ascending=True)

    if len(fortnightly_filtered.index) > 0:
        # Add columns to plot actual and normal rainfall
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Actual Rainfall Map")
            # Plot district shapefile with daily rainfall statistics
            add_fortnight_actual_vector(district_gdf, fortnightly_filtered, fortnightly_intervals)
        with col2:
            st.subheader("Normal Rainfall Map")
            # Plot district shapefile with daily rainfall statistics
            add_fortnight_normal_vector(district_gdf, fortnightly_filtered, fortnightly_intervals)
        plot_bars(fortnightly_filtered)

        # Export daily data as CSV
        out_fn = fortnightly_filtered['stname'].iloc[0] + '_' + fortnightly_filtered['Fortnight'].iloc[0] + '.csv' # Output filename
        dload_data = fortnightly_filtered.to_csv(index=False).encode("utf-8")
        export_btn = st.sidebar.download_button('Export CSV', data=dload_data, file_name=out_fn, mime='text/csv', key='export_fortnightly')
else:
    st.subheader(':blue[Monthly Rainfall Statistics]')
    month = st.sidebar.selectbox('Select Month:', month_dict) # Add Month widget to sidebar
    month_no = month_dict.get(month)    # Get month no. from month dictinary

    # Filter monthly rainfall data of selected state
    monthly_filtered = monthly_df.loc[(monthly_df['Date'] == monthly_dates_dict.get(month_no)) & (monthly_df['stname'] == st.session_state.state)]
    monthly_filtered = monthly_filtered.sort_values(by=['dtname'], ascending=True)

    if len(monthly_filtered.index) > 0:
        # Add columns to plot actual and normal rainfall
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Actual Rainfall Map")
            # Plot district shapefile with daily rainfall statistics
            add_actual_vector(district_gdf, monthly_filtered, monthly_intervals)
        with col2:
            st.subheader("Normal Rainfall Map")
            # Plot district shapefile with daily rainfall statistics
            add_normal_vector(district_gdf, monthly_filtered, monthly_intervals)
        plot_bars(monthly_filtered)

        # Export daily data as CSV
        monthly_filtered = monthly_filtered.drop(columns='Date')
        monthly_filtered['Month'] = month
        out_fn = monthly_filtered['stname'].iloc[0] + '_' + month + '.csv' # Output filename
        dload_data = monthly_filtered.to_csv(index=False).encode("utf-8")
        export_btn = st.sidebar.download_button('Export CSV', data=dload_data, file_name=out_fn, mime='text/csv', key='export_monthly')
        st.write(out_fn)


