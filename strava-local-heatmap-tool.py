## Strava Local Heatmap Tool
# Last update: 2022-04-12


# Erase all declared global variables
globals().clear()

# Import packages
import os
import glob
import gzip
import datetime
import time
import shutil
import webbrowser

import folium
from geopy.geocoders import Nominatim
import pandas as pd
import janitor
import numpy as np
import sweat
from plotnine import *

# Set working directory to user's 'Downloads/Strava' folder
os.chdir(os.path.join(os.path.expanduser('~'), r'Downloads/Strava'))

# Create output folder inside Strava folder
os.makedirs('output', exist_ok=True)



###########
# Functions
###########

## Extract .gz files
# Adapted from: https://gist.github.com/kstreepy/a9800804c21367d5a8bde692318a18f5
def gz_extract(directory):
    extension = '.gz'
    current_directory = os.getcwd()
    os.chdir(directory)
    for item in os.listdir(directory): # Loop through items in dir
      if item.endswith(extension): # Check for ".gz" extension
          gz_name = os.path.abspath(item) # Get full path of files
          file_name = (os.path.basename(gz_name)).rsplit('.',1)[0] # Get file name for file within
          with gzip.open(gz_name,'rb') as f_in, open(file_name,'wb') as f_out:
              shutil.copyfileobj(f_in, f_out)
          os.remove(gz_name) # Delete zipped file
    os.chdir(current_directory)



## Remove leading first line blank spaces of .tcx activity files
def tcx_lstrip():

    ## List of .tcx files to be imported
    activities_files_list = glob.glob(r'activities/*.tcx')
    
    ## Remove leading spaces from first row
    for index in activities_files_list:
    
        with open(file=index, mode='r') as f:
            lines = f.readlines()
            lines[0] = lines[0].lstrip()

        with open(file=index, mode='w') as f:
            f.writelines(lines)



## Import .fit/.gpx/.tcx activity files into a dataframe
def import_activities():

    ## Create or import global variables
    global activities_points
    
    ## List of files to be imported
    activities_files_list = glob.glob(r'activities/*.fit')
    activities_files_list.extend(glob.glob(r'activities/*.gpx'))
    activities_files_list.extend(glob.glob(r'activities/*.tcx'))

    ## Create empty dataframe
    activities_points = pd.DataFrame(dtype=object)

    ## Import activities
    for index in activities_files_list:
        
        try:
            # Import file and convert to dataframe
            df = sweat.read_file(fpath=index)
            
            # Create new column with the name of the file
            df['filename'] = index
            df['filename'] = df['filename'].str.replace(r'^activities\\', r'activities/', regex=True)
            
            # Concatenate dataframe
            activities_points = pd.concat([activities_points, df])
        
        except:
            pass
        
    # Convert index to column
    activities_points.reset_index(level=0, inplace=True)
    
    # Select and rearrange columns
    activities_points = activities_points.filter(['datetime', 'filename', 'latitude', 'longitude'])
    
    # Rearrange rows
    activities_points = activities_points.sort_values(by=['datetime', 'filename'], ignore_index=True)
    
    ## Get elapsed time (in seconds)
    #activities_points['elapsed_time'] = activities_points.groupby('filename', as_index=False)['datetime'].transform(lambda row: (row.max() - row.min()).total_seconds())
    
    # Remove rows without latitude/longitude
    activities_points = activities_points[activities_points['latitude'].notna()]

    ## Return objects
    return activities_points



## Get geolocation for .fit/.gpx/.tcx activity files given the start recorded point (first non-missing latitude/longitude)
def get_geolocation():
    
    ## Create or import global variables
    global activities_points
    global activities_location
    
    activities_location = activities_points
    
    # Get first row for each group of filename
    activities_location = activities_location.groupby('filename', as_index=False).first()

    ## Get location
    geolocator = Nominatim(user_agent='http')
    
    activities_location['location'] = activities_location.apply(lambda row: geolocator.reverse('{}, {}'.format(row['latitude'], row['longitude']), language='en') if pd.notna(row['latitude']) else np.nan, axis=1)
    activities_location['country'] = activities_location.apply(lambda row: row['location'].raw.get('address').get('country') if pd.notna(row['location']) else np.nan, axis=1)
    activities_location['state'] = activities_location.apply(lambda row: row['location'].raw.get('address').get('state') if pd.notna(row['location']) else np.nan, axis=1)
    activities_location['city'] = activities_location.apply(lambda row: row['location'].raw.get('address').get('city') if pd.notna(row['location']) else np.nan, axis=1)
    activities_location['postal_code'] = activities_location.apply(lambda row: row['location'].raw.get('address').get('postcode') if pd.notna(row['location']) else np.nan, axis=1)

    # Rename columns
    activities_location = activities_location.rename(columns = {'country': 'activity_country', 'state': 'activity_state', 'city': 'activity_city', 'postal_code': 'activity_postal_code', 'latitude': 'activity_latitude', 'longitude': 'activity_longitude'})
    
    # Select and rearrange columns
    activities_location = activities_location.filter(['filename', 'activity_country', 'activity_state', 'activity_city', 'activity_postal_code', 'activity_latitude', 'activity_longitude'])

    ## Return objects
    return activities_location



## Copy activity list files to output\activities folder
def copy_activities(list):
    os.makedirs(r'output/activities', exist_ok=True)
    for filename in ('activities//'+list).tolist():
        shutil.copy(filename, r'output/activities')




########################
# Import activity points
########################

## Extract .gz files
gz_extract(os.path.join(os.getcwd(), r'activities'))

## Remove leading first line blank spaces of .tcx activity files
tcx_lstrip()

## Import .fit/.gpx/.tcx activity files into a dataframe
import_activities()

## Save activities_points
activities_points.to_csv(path_or_buf = r'output/activities_points.csv', sep = ',', index = False, encoding='utf-8')

## Load activities_points
# activities_points = pd.read_csv(r'output/activities_points.csv', sep = ',')

# Get geolocation for .fit/.gpx/.tcx activity files given the start recorded point (first non-missing latitude/longitude)
get_geolocation()

# Test for unique 'filename'
pd.Series(activities_location['filename']).is_unique




################################
# Import Strava's activity files
################################

## Strava's activities column definitions
# https://developers.strava.com/docs/reference/#api-models-DetailedActivity

# elapsed_time, moving_time: seconds
# distance, elevation_gain, elevation_loss: meters
# max_speed, average_speed: meters/second

## Import Strava activities
activities = pd.read_csv(r'activities.csv')
activities = activities.clean_names()

## Clean 'filename' variable
#activities['filename'] = activities['filename'].str.replace(r'^activities/', r'', regex=True)
activities['filename'] = activities['filename'].str.replace(r'\.gz$', r'', regex=True)

## Get activities location
activities = activities.merge(activities_location.filter(['filename', 'activity_country', 'activity_state', 'activity_city', 'activity_postal_code', 'activity_latitude', 'activity_longitude']), how='left', on=['filename'])

## Delete objects
del activities_location

## Select and rearrange columns
activities = activities.filter(['activity_date', 'activity_id', 'filename', 'from_upload', 'activity_country', 'activity_state', 'activity_city', 'activity_postal_code', 'activity_latitude', 'activity_longitude', 'activity_type', 'commute_1', 'activity_name', 'activity_description', 'activity_gear', 'elapsed_time', 'moving_time', 'distance_1', 'max_speed', 'average_speed', 'elevation_gain', 'elevation_loss', 'elevation_low', 'elevation_high', 'max_grade', 'average_grade', 'grade_adjusted_distance', 'max_cadence', 'average_cadence', 'max_heart_rate', 'average_heart_rate', 'max_watts', 'average_watts', 'calories', 'relative_effort', 'weighted_average_power', 'power_count', 'perceived_exertion', 'perceived_relative_effort', 'total_weight_lifted', 'athlete_weight', 'bike_weight'])

## Rename columns
activities = activities.rename(columns={'distance_1': 'distance', 'commute_1': 'commute'})

## Change dtypes
activities = activities.to_datetime('activity_date')

## Convert elapsed_time and moving_time from seconds to minutes
activities = activities.assign(elapsed_time=activities['elapsed_time']/60,
    moving_time=activities['moving_time']/60)

## Convert max_speed and average_speed from m/s to km/h
activities = activities.assign(max_speed=activities['max_speed']*3.6,
    average_speed=activities['average_speed']*3.6)

## Rearrange rows
activities = activities.sort_values(by=['activity_date', 'activity_type'], ignore_index=True)

## Save activities
activities.to_csv(path_or_buf=r'output/activities.csv', sep=',', index=False, encoding='utf-8')

## Load activities
# activities=pd.read_csv(r'output/activities.csv', sep=',')



### Tests

## Check for activities without activity_gear
(activities[activities['activity_gear'].isnull()]
    .groupby('activity_type')
    .agg(count=('activity_id', 'nunique'))
    .reset_index())

## Check for activity_name inconsistencies
activities[activities['activity_name'].str.contains('  ') | activities['activity_name'].str.contains(' $') | activities['activity_name'].str.contains('^ ')]
activities[activities['activity_name'].str.contains('[^\\s]-') | activities['activity_name'].str.contains('-[^\\s]')]

## Check for distinct values for activity_name separated by a hyphen
pd.unique(activities[activities['activity_type'] == 'Ride']['activity_name'].str.split(pat=' - ', expand=True).stack()).tolist()



### Summary

## Count of activities by type
(activities
    .groupby('activity_type')
    .agg(count=('activity_id', 'nunique'))
    .reset_index())


## Runs overview per month (distance in km)
(activities[activities['activity_type'] == 'Run']
    .groupby(pd.Grouper(key='activity_date', freq='M'))
    .agg(count=('activity_id', 'nunique'), distance=('distance', lambda x: x.sum()/1000))
    .reset_index()
    .sort_values(by=['activity_date'], ascending=True))


## Strava yearly overview cumulative (Plot)
strava_yearly_overview = activities[activities['activity_type'] == 'Ride']
strava_yearly_overview = strava_yearly_overview[(strava_yearly_overview['activity_date'] >= '2017-01-01') & (strava_yearly_overview['activity_date'] <= '2021-12-31')]
strava_yearly_overview = strava_yearly_overview.assign(distance = strava_yearly_overview['distance']/1000,
    year = strava_yearly_overview['activity_date'].dt.year,
    distance_cumulative = lambda x: x.groupby(pd.Grouper(key = 'activity_date', freq = 'Y'))['distance'].transform('cumsum'),
    activity_date = strava_yearly_overview['activity_date'].dt.dayofyear)

(ggplot(strava_yearly_overview, aes(x='activity_date', y='distance_cumulative', group='year', color='factor(year)')) +
    geom_line() +
    scale_color_brewer(palette=1) +
	theme_minimal() +
    #theme(legend_position='bottom') +
	labs(title='Cumultative Distance (KM)', y='Distance (KM)', x='Day of Year', color='Year'))




#########
# Heatmap
#########

### Filter activities
activities_filtered = activities

## Remove activities without latitude/longitude points
activities_filtered = activities_filtered[activities_filtered['filename'].notna()]

## Filter activities by type
#activities_filtered = activities_filtered[activities_filtered['activity_type'].isin(['Ride', 'Run'])]

## Filter activities by state
#activities_filtered = activities_filtered[activities_filtered['activity_state'].isin(['Baden-Württemberg', 'Bavaria', 'Lower Austria', 'Salzburg', 'Tyrol', 'Vorarlberg'])]


## Filter activities inside a region boundary rectangle

# Munich
#(latitude_2_2, longitude_2_2) = 48.23164463142422, 11.71709016935086 # Top right boundary
#(latitude_2_1, longitude_1_2) = 48.226119172326904, 11.452182735005932 # Top left boundary
#(latitude_1_1, longitude_1_1) =  48.085113214484245, 11.402243511405025 # Bottom left boundary
#(latitude_1_2, longitude_2_1) = 48.069684876663985, 11.768838030115262 # Bottom right boundary

# Greater Munich
#(latitude_2_2, longitude_2_2) = 48.40320310285019, 11.82558556949217  # Top right boundary
#(latitude_2_1, longitude_1_2) = 48.392468860536404, 11.308237208617681 # Top left boundary
#(latitude_1_1, longitude_1_1) = 47.90088906713716, 11.070349346995181  # Bottom left boundary
#(latitude_1_2, longitude_2_1) = 47.8609192626143, 12.110508804713536 # Bottom right boundary

# Southern Bavaria
#(latitude_2_2, longitude_2_2) = 47.790061994764315, 12.269239237840651 # Top right boundary
#(latitude_2_1, longitude_1_2) = 47.7948920311783, 10.920373081783941 # Top left boundary
#(latitude_1_1, longitude_1_1) = 47.402331254142894, 10.977963609533084 # Bottom left boundary
#(latitude_1_2, longitude_2_1) = 47.43919436807553, 12.318763470661775 # Bottom right boundary

#activities_filtered = activities_filtered[activities_filtered['activity_latitude'].between(min(latitude_1_1, latitude_1_2), max(latitude_2_1, latitude_2_2))]
#activities_filtered = activities_filtered[activities_filtered['activity_longitude'].between(min(longitude_1_1, longitude_1_2), max(longitude_2_1, longitude_2_2))]


## Filter activities points given the filtered activities
activities_points_filtered = activities_points[activities_points['filename'].isin(activities_filtered['filename'])]
activities_points_filtered = activities_points_filtered.filter(['datetime', 'filename', 'latitude', 'longitude'])

## Test
len(activities_points_filtered['filename'].drop_duplicates()) == len(activities_filtered)

## Get activity_id and activity_type
activities_points_filtered = activities_points_filtered.merge(activities[['filename', 'activity_id', 'activity_type']], how='left', on='filename').drop('filename', axis=1)

## Test memory usage
activities_points_filtered.info(memory_usage='deep')

## Transform latitude/longitude to list
activities_points_filtered['point'] = list(zip(activities_points_filtered['latitude'], activities_points_filtered['longitude']))

# Copy filtered activity list files to output\activities folder
#copy_activities(list=activities_filtered['filename'])



### Heatmap

## Define activity_type colors
activity_colors = {'Hike':'#00AD43', # green
'Ride':'#FF5800', # orange
'Run':'#00A6FC', # blue
}


## Define map tile
tile = 'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}@2x.png'
#tile = 'https://a.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}@2x.png'
#tile = 'https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png'
#tile = 'https://a.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}@2x.png'
#tile = 'http://tile.stamen.com/terrain-background/{z}/{x}/{y}.png'
#tile = 'http://tile.stamen.com/toner-lite/{z}/{x}/{y}.png'
#tile = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}'
#tile = 'https://server.arcgisonline.com/ArcGIS/rest/services/Ocean_Basemap/MapServer/tile/{z}/{y}/{x}'


## Define Folium map settings
latitude = round(activities_points_filtered['latitude'].median(), 4)
longitude = round(activities_points_filtered['longitude'].median(), 4)
zoom_start = 12
line_weight = 1.0 # For activities spread throughout the map, use line_weight=2.0


## Create Folium map
activities_map = folium.Map(tiles=tile, attr='tile', location=[latitude, longitude], zoom_start=zoom_start)
folium.LayerControl().add_to(activities_map)


## Plot activities into Folium map
# Adapted from: https://github.com/andyakrn/activities_heatmap
for activity_type in activities_points_filtered['activity_type'].unique():
    df_activity_type = activities_points_filtered[activities_points_filtered['activity_type']==activity_type]
    
    for activity in df_activity_type['activity_id'].unique():
            date = df_activity_type[df_activity_type['activity_id']==activity]['datetime'].dt.date.iloc[0]
             
            points = tuple(df_activity_type[df_activity_type['activity_id']==activity]['point'])
            folium.PolyLine(points, color=activity_colors[activity_type],
                            weight=line_weight,
                            opacity=0.6,
                            control=True,
                            name=activity_type,
                            popup=activity_type+' (ID: '+'<a href=https://www.strava.com/activities/'+str(activity)+'>'+str(activity)+'</a>), '+str(date),
                            tooltip=activity_type,
                            smooth_factor=1.0,
                            overlay=True).add_to(activities_map)


## Save to .html file
activities_map.save('output/activities_map.html')
webbrowser.open('output/activities_map.html')



#### Summary


# Total activities
print(activities_filtered.shape[0])

# Total distance (in km)
print(round(activities_filtered['distance'].sum()/1000, 1))

# Total moving time (in days, hours, minutes, seconds)
print(datetime.timedelta(seconds=(activities_filtered.assign(moving_time=activities_filtered['moving_time']*60)['moving_time']).sum()))

# Total elevation gain
print(round(activities_filtered['elevation_gain'].sum()/1000, 1))

# Longest activity (in m)
print(activities_filtered[activities_filtered['distance']==activities_filtered['distance'].max()].filter(['activity_date', 'distance']))

# Max speed (km/h)
print(activities_filtered['max_speed'].max())

# Average speed (km/h)
print(activities_filtered['average_speed'].mean())
