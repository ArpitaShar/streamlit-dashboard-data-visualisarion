import streamlit as st

def page_Crop_health_bulletin():
    st.title("🛰️ Crop_health_bulletin Page 📊")
    st.write("Dashboard which can be used to visualize NDVI and LSWI datasets from MODIS (MOD13Q1 - 16-day composite) and their profiles with corresponding historical datasets. The visualization can be performed for different administrative boundaries (subdistricts of different states of India).")

def page_weather_Information():
    st.title("🌧️ weather_Information Page 📈")
    st.write("Dashboard which can be used to visualize rainfall data from IMD (0.25 x 0.25 degree) across different districts within a state. The data can be visualized for different time-periods viz. Daily, Weekly, Fortnightly and Monthly.")

# Dictionary to map page names to functions
pages = {
    "🛰️Crop Health Information": page_Crop_health_bulletin,
    "🌧️Weather Information": page_weather_Information,
    }

def main():
    st.title("🌐Analytical overview for Crop🌾")

    # Add a radio button to select the page
    page = st.radio("Select a page:", tuple(pages.keys()))

    # Call the function corresponding to the selected page
    pages[page]()

if __name__ == "__main__":
    main()

    # pages/page1.py

def app():
    st.header("Crop_health_bulletin")
    st.write("This is the content of Crop_health_bulletin.")

def app():
    st.header("weather_Information")
    st.write("This is the content of Page weather_Information.")







    

    


    












    




