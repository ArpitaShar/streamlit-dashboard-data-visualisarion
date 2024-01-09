"""
Created on Thu Apr 20 10:53:45 2023

@author: Arpita Sharma
"""

# import libraries
import streamlit as st
import os
import glob
import leafmap.foliumap as leafmap
import geopandas as gpd
from geopandas import GeoDataFrame
import numpy as np
import pandas as pd
import rasterstats
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling, calculate_default_transform
import numpy as np
from datetime import datetime
import plotly.graph_objects as go


st.set_page_config(layout='wide', initial_sidebar_state='expanded')

st.title("🛰️ MODIS Data Analysis Platform (MODAP) 📊")

modis_data_dir = r"F:\Automation_Working\modis_data"
agrimask_data = r"F:\Temp\Automation\agrimask\india_agrimask.vrt"
shapefile_path = r"F:\Temp\Automation\shapefiles\Subdistrict_Shapefile.shp"
temp_dir = r"F:\Temp\temp_files\test"

# year_list = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
# month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

year_list = [2022, 2023]
indices_list = ['NDVI', 'LSWI']

fn_dict = {'Jan 01 - Jan 16' : 1,
           'Jan 17 - Feb 01' : 2,
           'Feb 02 - Feb 17' : 3,
           'Feb 18 - Mar 05' : 4,
           'Mar 06 - Mar 21' : 5,
           'Mar 22 - Apr 06' : 6,
           'Apr 07 - Apr 22' : 7,
           'Apr 23 - May 08' : 8,
           'May 09 - May 24' : 9,
           'May 25 - Jun 09' : 10,
           'Jun 10 - Jun 25' : 11,
           'Jun 26 - Jul 11' : 12,
           'Jul 12 - Jul 27' : 13,
           'Jul 28 - Aug 12' : 14,
           'Aug 13 - Aug 28' : 15,
           'Aug 29 - Sep 13' : 16,
           'Sep 14 - Sep 29' : 17,
           'Sep 30 - Oct 15' : 18,
           'Oct 16 - Oct 31' : 19,
           'Nov 01 - Nov 16' : 20,
           'Nov 17 - Dec 02' : 21,
           'Dec 03 - Dec 20' : 22,
           'Dec 21 - Dec 31' : 23,
           } 

julian_dates_dict = {1 : '001',
                     2 : '017',
                     3 : '033',
                     4 : '049',
                     5 : '065',
                     6 : '081',
                     7 : '097',
                     8 : '113',
                     9 : '129',
                     10 : '145',
                     11 : '161',
                     12 : '177',
                     13 : '193',
                     14 : '209',
                     15 : '225',
                     16 : '241',
                     17 : '257',
                     18 : '273',
                     19 : '289',
                     20 : '305',
                     21 : '321',
                     22 : '337',
                     23 : '353'
                     }

# Add Indices selectbox to the sidebar:
if 'crop_index' not in st.session_state:
    st.session_state.crop_index = 'None'
crop_index = st.sidebar.selectbox('Select Indices', indices_list)
st.session_state.crop_index = crop_index

# Add Year selectbox to the sidebar:
year = st.sidebar.selectbox('Select Year', year_list)

# Add Month selectbox to the sidebar:
# month = st.sidebar.selectbox('Select Month', month_list)

# Add Fortnight selectbox to the sidebar:
fortnight = st.sidebar.selectbox('Select Fortnight', fn_dict)


@st.cache_data
def read_data():
     # Import shapefile
    shp_gdf = gpd.read_file(shapefile_path)
    shp_gdf = shp_gdf.replace('\n','', regex=True)

    return shp_gdf

# Function to filter and select NDVI raster
def filter_ndvi(year):
    ndvi_dir = os.path.join(modis_data_dir, 'NDVI\\{}'.format(year))
    ndvi_files = glob.glob(ndvi_dir + '/*.tif')

    date = 'doy' + str(year) + julian_dates_dict.get(fn_dict.get(fortnight))
    ndvi_filtered_list = list(filter(lambda x: date in x, ndvi_files))
    if len(ndvi_filtered_list) == 0:
        st.error('NDVI data not available for given dates')
    else:
        filename = ndvi_filtered_list[0]

    return filename

# Function to filter and select LSWI raster
def filter_lswi(year):
    lswi_dir = os.path.join(modis_data_dir, 'LSWI\\{}'.format(year))
    lswi_files = glob.glob(lswi_dir + '/*.tif')

    date = 'doy' + str(year) + julian_dates_dict.get(fn_dict.get(fortnight))
    lswi_filtered_list = list(filter(lambda x: date in x, lswi_files))
    if len(lswi_filtered_list) == 0:
        st.error('LSWI data not available for given dates')
    else:
        filename = lswi_filtered_list[0]

    return filename

# Function to mask out non-agri area and resample modis data
def mask_raster(srcRaster, agriMaskRaster, shp_gdf, outRaster):
    # This function clips source and mask raster to geodataframe extent,
    # resamples source data to mask raster's pixel size and then
    # masks out pixels from resampled source raster using mask raster
    
    # Clip MODIS raster using geodataframe geometry
    with rasterio.open(srcRaster) as modisData:        
        clipped_modis, out_transform_modis = mask(modisData, shp_gdf.geometry, crop=True)
        
        # Update metadata
        modis_meta = modisData.meta.copy()
        modis_meta.update({
            "driver": "GTiff",
            "height": clipped_modis.shape[1],
            "width": clipped_modis.shape[2],
            "transform": out_transform_modis
        })
        
    # Save output to memory
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**modis_meta) as dst_modis:
            dst_modis.write(clipped_modis)
            
            
            # Clip AgriMask raster using geodataframe geometry
            with rasterio.open(agriMaskRaster) as agrimask:              
                clipped_agrimask, out_transform_agrimask = mask(agrimask, shp_gdf.geometry, crop=True)
                
                # Update metadata
                agrimask_meta = agrimask.meta.copy()
                agrimask_meta.update({
                    "driver": "GTiff",
                    "height": clipped_agrimask.shape[1],
                    "width": clipped_agrimask.shape[2],
                    "transform": out_transform_agrimask
                })
                
            # Save output to memory
            with rasterio.MemoryFile() as memfile:
                with memfile.open(**agrimask_meta) as dst_agrimask:
                    dst_agrimask.write(clipped_agrimask)

                    # Resmaple MODIS data to agrimask's resolution
                    src_transform = dst_modis.transform
                    dst_crs = dst_agrimask.crs
                        
                    # calculate the output transform matrix
                    dst_transform, dst_width, dst_height = calculate_default_transform(
                        dst_modis.crs,     # input CRS
                        dst_crs,     # output CRS
                        dst_agrimask.width,   # input width
                        dst_agrimask.height,  # input height 
                        *dst_agrimask.bounds,  # unpacks input outer boundaries (left, bottom, right, top)
                    )
                
                    # set properties for output
                    dst_kwargs = dst_modis.meta.copy()
                    dst_kwargs.update({"crs": dst_crs,
                                        "transform": dst_transform,
                                        "width": dst_width,
                                        "height": dst_height,
                                        "nodata": 0
                                        #'dtype': 'uint8'
                                        })

                    # Save output to memory
                    with rasterio.MemoryFile() as memfile:
                        with memfile.open(**dst_kwargs) as modis_resampled:
                        # iterate through bands and write using reproject function
                            for i in range(1, dst_modis.count + 1):
                                reproject(
                                    source = rasterio.band(dst_modis, i),
                                    destination = rasterio.band(modis_resampled, i),
                                    src_transform = src_transform,
                                    src_crs = dst_modis.crs,
                                    dst_transform = dst_transform,
                                    dst_crs = dst_crs,
                                    resampling = Resampling.nearest)
                                
                            
                            # ### Mask out pixels in resampled modis using agrimask
                            out_meta = modis_resampled.meta.copy()  # Copy resampled modis data's metadata
                            resampled_arr = modis_resampled.read(1) # Read resmapled modis data as array
                            agrimask_arr = dst_agrimask.read(1)     # Read agrimask data as array
                            if st.session_state.crop_index == 'NDVI':
                                # out_raster_fn = os.path.join(temp_dir, (shp_gdf['sdtname'].iloc[0] + '_NDVI_'+ datetime.now().strftime("%d-%m-%Y_%Hh%Mm%Ss") +'.tif'))
                                masked_arr = resampled_arr*agrimask_arr*0.0001 # Masking out non crop areas where pixels value in agrimask data is zero
                            else:
                                # out_raster_fn = os.path.join(temp_dir, (shp_gdf['sdtname'].iloc[0] + '_LSWI_'+ datetime.now().strftime("%d-%m-%Y_%Hh%Mm%Ss") +'.tif'))
                                masked_arr = resampled_arr*agrimask_arr # Masking out non crop areas where pixels value in agrimask data is zero

                            out_meta.update({"dtype":'float32'})    # Update data type to float32

                            # Write the masked raster to disk
                            with rasterio.open(outRaster, "w", **out_meta) as dst_masked:
                                dst_masked.write(masked_arr, 1)

    return outRaster

# Function to mask rasters(time-series) for zonal statistics calculation
def mask_raster_batch(rasterList, agriMaskRaster, shp_gdf):
    masked_rasters_ls = []

    # convert each row into individual dataframe
    for index, row in shp_gdf.iterrows():
        gdf = gpd.GeoDataFrame(row).T
        for i in rasterList:
            temporal_year = i[i.index('doy')+3:i.index('doy')+7]
            if st.session_state.crop_index == 'NDVI':
                out_raster_fn = os.path.join(temp_dir, (gdf['sdtname'].iloc[0] + '_' + temporal_year + '_NDVI_' + datetime.now().strftime("%d-%m-%Y_%Hh%Mm%Ss") +'.tif'))
            else:
                out_raster_fn = os.path.join(temp_dir, (gdf['sdtname'].iloc[0] + '_' + temporal_year + '_LSWI_' + datetime.now().strftime("%d-%m-%Y_%Hh%Mm%Ss") +'.tif'))

            masked_raster = mask_raster(i, agriMaskRaster, gdf, out_raster_fn)
            masked_rasters_ls.append(masked_raster)

    return masked_rasters_ls

# Function to calculate CV for zonal statistics of a given feature
def calculate_cv(x):
    cv =  np.std(x) / np.mean(x)
    return cv

# Function to calculate zonal statistics for a given feature
def calculate_zonal_stats(rasters_list, shp_gdf):

    # Create an empty dataframe to store zonal statistics
    zonal_df = pd.DataFrame(columns=['Name', 'Year', 'min', 'max', 'CV'])
    
    # convert each row into individual dataframe
    for index, row in shp_gdf.iterrows():
        shapefile_gdf = gpd.GeoDataFrame(row).T
    
        default_crs = 'EPSG:4326'
        
        # Set CRS to the default if needed.
        if shapefile_gdf.crs is None:
            shapefile_gdf = shapefile_gdf.set_crs(default_crs)
        
        # Transform to a Raster's CRS.
        features = shapefile_gdf.to_crs(default_crs)
        features.reset_index(inplace = True)

        # Loop over each raster file and calculate zonal statistics
        matching_raster_ls = [s for s in rasters_list if shapefile_gdf['sdtname'].iloc[0] in s]
        for raster_file in matching_raster_ls:
            temporal_year = raster_file[-34:-30]
            stats = rasterstats.zonal_stats(features, raster_file,
                                            stats="min max mean",
                                            nodata=0,
                                            add_stats = {'CV':calculate_cv},
                                            all_touched=True)
            values_dict = stats[0]
            values_dict['Year'] = temporal_year
            values_dict['Name'] = shapefile_gdf['sdtname'].iloc[0]
            zonal_df = zonal_df.append(values_dict, ignore_index=True)
                
    if st.session_state.crop_index == 'NDVI':
        zonal_df.rename(columns = {'min' : 'NDVI' + '_min',
                                'max' : 'NDVI' + '_max',
                                'mean' : 'NDVI' + '_mean',
                                'CV' : 'NDVI' + '_CV'}, inplace = True)
    else:
        zonal_df.rename(columns = {'min' : 'LSWI' + '_min',
                                'max' : 'LSWI' + '_max',
                                'mean' : 'LSWI' + '_mean',
                                'CV' : 'LSWI' + '_CV'}, inplace = True)        

    return zonal_df

# Function to plot zonal statistics of selected SDT
def plot_historical(raster_list, gdf):

    # Calculate zonal stats
    df = calculate_zonal_stats(raster_list, gdf)
    
    plot_title = ' Mean ' + st.session_state.crop_index + ' Profile (Historical)'
    
    # Set y-axis based on the indices selected
    if st.session_state.crop_index == 'NDVI':
        y_axis = 'NDVI_mean'
    else:
        y_axis = 'LSWI_mean'
    # Get Unique values of sdt names
    unique_values = df['Name'].unique().tolist()
    unique_values.sort()

    # Create traces (lines) for each unique sdt
    traces = []
    for value in unique_values:
        trace = go.Scatter(x=df['Year'], y=df[df['Name'] == value][y_axis], name=value,
                           line=dict(width=3), marker_size = 12)
        traces.append(trace)
    
    layout = go.Layout(title=plot_title, xaxis=dict(title='Year'), yaxis=dict(title=st.session_state.crop_index),
                        yaxis_range=[0,1], font_size=18, width=800, height=600)
    fig = go.Figure(data=traces, layout=layout)
    st.plotly_chart(fig, theme=None, use_container_width=True)


def app():

    ndvi_rasters_ls = [filter_ndvi(year), filter_ndvi(year-1), filter_ndvi(year-2), filter_ndvi(year-3), filter_ndvi(year-4), filter_ndvi(year-5)]
    lswi_rasters_ls = [filter_lswi(year), filter_lswi(year-1), filter_lswi(year-2), filter_lswi(year-3), filter_lswi(year-4), filter_lswi(year-5)]

    # # Import shapefile
    Shapefile = read_data()

    col1, col2, col3, col4 = st.columns(4)

    #data management 1
    with col1:
        stname_option = Shapefile['stname'].unique().tolist()
        stname_option.sort()
        stname= st.selectbox('STATE', stname_option, 0)
        Shapefile = Shapefile[Shapefile['stname']== stname]

        #data management 2
        with col2:
            dtname_option = Shapefile['dtname'].unique().tolist()
            dtname_option.sort()
            dtname = st.selectbox('DISTRICT', dtname_option, 0)
            Shapefile = Shapefile[Shapefile['dtname']== dtname]

            #data management 3
            with col3:
                sdtname_option = Shapefile['sdtname'].unique().tolist()
                sdtname_option.sort()
                sdtname = st.multiselect('SUBDISTRICT', sdtname_option)
                Shapefile = Shapefile[Shapefile['sdtname'].isin(sdtname)]
                gdf = GeoDataFrame(Shapefile, crs="EPSG:4326", geometry = Shapefile.geometry)


    st.sidebar.header('Plotting Options:')
    # zonal_stats_btn = st.button('Calculate Zonal Statistics', key='zonal_stats')
    plot_raster_btn = st.sidebar.button('Plot ' + st.session_state.crop_index + ' Image', key='plot_raster_btn')
    plot_zonal_btn = st.sidebar.button('Plot ' + st.session_state.crop_index + ' Profile', key='plot_zonal_btn')

    st.sidebar.header('Export Options:')
    export_zonal_btn = st.sidebar.button('Export Zonal Statistics', key='export_zonal')
    export_raster_btn = st.sidebar.button('Export ' + st.session_state.crop_index + ' raster', key='export_raster')

    gdf.reset_index(inplace = True)
    
    m = leafmap.Map(center=[20.5937, 78.9629], zoom=4)
    m.add_basemap(google_map="HYBRID", show=False)


    if plot_zonal_btn:
        if st.session_state.crop_index == 'NDVI':
            raster_ls = mask_raster_batch(ndvi_rasters_ls, agrimask_data, gdf)
            plot_historical(raster_ls, gdf)
        else:
            raster_ls = mask_raster_batch(lswi_rasters_ls, agrimask_data, gdf)
            plot_historical(raster_ls, gdf)

    if plot_raster_btn:
        if st.session_state.crop_index == 'NDVI':
            ndvi_fn_current = filter_ndvi(year)
            if len(gdf.index) == 1:
                raster_fn = os.path.join(temp_dir, (gdf['sdtname'].iloc[0] + '_NDVI_'+ datetime.now().strftime("%d-%m-%Y_%Hh%Mm%Ss") +'.tif'))
            else:
                raster_fn = os.path.join(temp_dir, (gdf['dtname'].iloc[0] + '_NDVI_'+ datetime.now().strftime("%d-%m-%Y_%Hh%Mm%Ss") +'.tif'))
            ndvi_image = mask_raster(ndvi_fn_current, agrimask_data, gdf, raster_fn)
            m.add_raster(ndvi_image, band=1, palette='RdYlGn', vmin=0, vmax=1, nodata=-0.3, layer_name='NDVI_' + str(year))
            colorbar = "https://i.postimg.cc/8kYL0f9P/MODIS-NDVI.png"
            m.add_image(colorbar, position='bottomright') 
        else:
            lswi_fn_current = filter_lswi(year)
            if len(gdf.index) == 1:
                raster_fn = os.path.join(temp_dir, (gdf['sdtname'].iloc[0] + '_LSWI_'+ datetime.now().strftime("%d-%m-%Y_%Hh%Mm%Ss") +'.tif'))
            else:
                raster_fn = os.path.join(temp_dir, (gdf['dtname'].iloc[0] + '_LSWI_'+ datetime.now().strftime("%d-%m-%Y_%Hh%Mm%Ss") +'.tif'))
            lswi_image = mask_raster(lswi_fn_current, agrimask_data, gdf, raster_fn)
            m.add_raster(lswi_image, band=1, palette='Blues', vmin=0, vmax=1, nodata=0, layer_name='LSWI_' + str(year))
            colorbar = "https://i.postimg.cc/g2myVF1v/MODIS-LSWI.png"
            m.add_image(colorbar, position='bottomright')

    if export_zonal_btn:
        if st.session_state.crop_index == 'NDVI':
            raster_ls = mask_raster_batch(ndvi_rasters_ls, agrimask_data, gdf)
            export_df = calculate_zonal_stats(raster_ls, gdf)
            if len(gdf.index) == 1:
                csv_fn = os.path.join(temp_dir, (gdf['sdtname'].iloc[0] + '_zonal_stats_NDVI.csv'))
            else:
                csv_fn = os.path.join(temp_dir, (gdf['dtname'].iloc[0] + '_zonal_stats_NDVI.csv'))
            export_df.to_csv(csv_fn)
        else:
            raster_ls = mask_raster_batch(lswi_rasters_ls, agrimask_data, gdf)
            export_df = calculate_zonal_stats(raster_ls, gdf)
            if len(gdf.index) == 1:
                csv_fn = os.path.join(temp_dir, (gdf['sdtname'].iloc[0] + '_zonal_stats_LSWI.csv'))
            else:
                csv_fn = os.path.join(temp_dir, (gdf['dtname'].iloc[0] + '_zonal_stats_LSWI.csv'))
            export_df.to_csv(csv_fn)
        st.success("Zonal statistics of " + gdf['dtname'].iloc[0] + " successfully exported (File is located at " + temp_dir + ").")

    if export_raster_btn:
        if st.session_state.crop_index == 'NDVI':
            ndvi_export = filter_ndvi(year)
            if len(gdf.index) == 1:
                raster_fn = os.path.join(temp_dir, (gdf['sdtname'].iloc[0] + '_NDVI_'+ datetime.now().strftime("%d-%m-%Y_%Hh%Mm%Ss") +'.tif'))
            else:
                raster_fn = os.path.join(temp_dir, (gdf['dtname'].iloc[0] + '_NDVI_'+ datetime.now().strftime("%d-%m-%Y_%Hh%Mm%Ss") +'.tif'))
            mask_raster(ndvi_export, agrimask_data, gdf, raster_fn)
            st.success("NDVI image of " + gdf['dtname'].iloc[0] + " exported successfully (File is located at " + temp_dir + ").")
        else:
            lswi_export = filter_lswi(year)
            if len(gdf.index) == 1:
                raster_fn = os.path.join(temp_dir, (gdf['sdtname'].iloc[0] + '_LSWI_'+ datetime.now().strftime("%d-%m-%Y_%Hh%Mm%Ss") +'.tif'))
            else:
                raster_fn = os.path.join(temp_dir, (gdf['dtname'].iloc[0] + '_LSWI_'+ datetime.now().strftime("%d-%m-%Y_%Hh%Mm%Ss") +'.tif'))
            mask_raster(lswi_export, agrimask_data, gdf, raster_fn)
            st.success("LSWI image of " + gdf['dtname'].iloc[0] + " exported successfully (File is located at " + temp_dir + ").")

    if len(gdf.index) > 0:
        m.add_gdf(gdf, style = {'fillOpacity': 0,
                                # 'fillColor': "#FF00FF",
                                'color': '#FE05EF',
                                'weight': 4,
                            },
                    layer_name = "Subdistrict")


    m.to_streamlit()

app()



# working 09 May 2023 (upto multi-line plot)    