<meta name='keywords' content='Strava, Heatmap, Local Heatmap, Folium, python'>

# Strava Local Heatmap Tool

## Description

This repository aims to be a multi feature tool for locally manipulating Strava's bulk export archive file. The main features are:
- Unzip compressed (.gz) activities files.
- Remove leading first line blank spaces of .tcx activities files for properly importing it (feature not yet available directly in the ```sweatpy``` package, see [here](https://github.com/GoldenCheetah/sweatpy/issues/99)).
- Import multiple .fit/.gpx/.tcx activities files at once (without the need of conversion) and create local highly customizable heatmaps with different colors by activity type with the use of the ```folium``` library.
- Increment your Strava activities metadata by adding country, state, city, postal code, latitude, longitude geolocation information for each activity given the start recorded point (first non-missing latitude/longitude).

Additionally, it is possible to apply a series of filters to select the desired activities before performing the heatmap, such as activities that started inside a bounding box (within 4 corner latitude/longitude points) or activities realized in specific countries or states.

Although similar projects already exist (see [here](#see-also)), some of the features implemented in this project were partial or non-existent.

## Output

Munich Heatmap (rides in orange; runs in blue)

<p align="center">
<img src="./media/heatmap_munich_1.png" alt="Heatmap Munich" width=800>
</p>

Vienna Heatmap (rides in orange; runs in blue)

<p align="center">
<img src="./media/heatmap_vienna_1.png" alt="Heatmap Vienna" width=800>
</p>

Map interaction (option to navigate through the map, click in a line and get an activity summary pop-up)

<p align="center">
<img src="./media/heatmap_popup_1.png" alt="Heatmap Pop-up" width=800>
</p>


# Usage

## Bulk export your Strava data

Strava's bulk export process documentation can be found [here](https://support.strava.com/hc/en-us/articles/216918437-Exporting-your-Data-and-Bulk-Export#Bulk).

Note: Please keep in mind that Strava's bulk export is language sensitive, i.e. the activities column labels will depend on users' defined language preferences. This project assumes that your bulk export was realized in ```English (US)```. To change the language, log in to [Strava](https://www.strava.com) and on the bottom right-hand corner of any page, select ```English (US)``` from the drop-down menu (more on this [here](https://support.strava.com/hc/en-us/articles/216917337-Changing-your-language-in-the-Strava-App)).

In essence, the process is as follows:
1. Log in to [Strava](https://www.strava.com).
2. Open the [Account Download and Deletion](https://www.strava.com/athlete/delete_your_account). Then press ```Request Your Archive``` button (Important: Don't press anything else on that page, particularly not the ```Request Account Deletion``` button).
3. Wait until Strava notifies you that your archive is ready via email. Download the archive file and unzip it to ```Downloads/Strava``` folder (or alternatively set a different working directory in the [strava-local-heatmap-tool.py](strava-local-heatmap-tool.py) code).


## Python dependencies

```.ps1
python -m pip install folium geopy pandas plotnine python-dateutil sweat
```


## Functions

### activities_import
```.py
activities_import()
```

#### Description
- Imports Strava activities assuming that the current working directory is the [Strava data bulk export](#bulk-export-your-strava-data).

#### Parameters
- None.

<br>

### activities_filter
```.py
activities_filter(activities_df=activities, activity_type=None, activity_state=None, bounding_box={'latitude_top_right': None, 'longitude_top_right': None, 'latitude_top_left': None, 'longitude_top_left': None, 'latitude_bottom_left': None, 'longitude_bottom_left': None, 'latitude_bottom_right': None, 'longitude_bottom_right': None})
```

#### Description
- Filter Strava activities DataFrame.


#### Parameters
- `activities_df`: Strava activities *DataFrame*. Imported from `activities_import()` function.
- `activity_type`: *str list*. If *None*, no activity type filter will be applied.
- `activity_state`: *str list*. If *None*, no state location filter will be applied.
- `bounding_box`: *dict*. If *None*, no bounding box will be applied.

Examples of `bounding_box`:

```.py
# Munich
bounding_box={
'latitude_top_right': 48.2316, 'longitude_top_right': 11.7170, # Top right boundary
'latitude_top_left': 48.2261, 'longitude_top_left': 11.4521, # Top left boundary
'latitude_bottom_left': 48.0851, 'longitude_bottom_left': 11.4022, # Bottom left boundary
'latitude_bottom_right': 48.0696, 'longitude_bottom_right': 11.7688 # Bottom right boundary
}
```

```.py
# Greater Munich
bounding_box={
'latitude_top_right': 48.4032, 'longitude_top_right': 11.8255, # Top right boundary
'latitude_top_left': 48.3924, 'longitude_top_left': 11.3082, # Top left boundary
'latitude_bottom_left': 47.9008, 'longitude_bottom_left': 11.0703, # Bottom left boundary
'latitude_bottom_right': 47.8609, 'longitude_bottom_right': 12.1105, # Bottom right boundary
}
```

```.py
# Southern Bavaria
bounding_box={
'latitude_top_right': 47.7900, 'longitude_top_right': 12.2692, # Top right boundary
'latitude_top_left': 47.7948, 'longitude_top_left': 10.9203, # Top left boundary
'latitude_bottom_left': 47.4023, 'longitude_bottom_left': 10.9779, # Bottom left boundary
'latitude_bottom_right': 47.4391, 'longitude_bottom_right': 12.3187, # Bottom right boundary
}
```

<br>

### heatmap
```.py
heatmap(activities_df=activities, activities_coordinates_df=activities_coordinates, activity_colors={'Hike': '#00AD43', 'Ride': '#FF5800', 'Run': '#00A6FC'}, map_tile='dark_all', map_zoom_start=12, line_weight=1.0, line_opacity=0.6, line_smooth_factor=1.0)
```

#### Description
- Create Heatmap based on inputted *activities* DataFrame.

#### Parameters
- `activities_df`: Strava activities *DataFrame*, default: *activities*. Imported from `activities_import()` function.
- `activities_coordinates_df`: Strava activities coordinates *DataFrame*, default: *activities_coordinates*. Imported from `activities_import()` function.
- `activity_colors`: *dict*, default: *{'Hike': '#00AD43', 'Ride': '#FF5800', 'Run': '#00A6FC'}*. Depending on how many distinct `activity_type` are contained in the `activities` DataFrame, more dictionaries objects need to be added.
- `map_tile`: *str*, options: *'dark_all'*, *'dark_nolabels'*, *'light_all'*, *'light_nolabels'*, *'terrain_background'*, *'toner_lite'* and *'ocean_basemap'*, default: *'dark_all'*.
- `map_zoom_start`: *int*, default: *12*. Initial zoom level for the map (for more details, check *zoom_start* parameter for [folium.folium.Map documentation](https://python-visualization.github.io/folium/modules.html#folium.folium.Map)).
- `line_weight`: *float*, default: *1.0*. Stroke width in pixels (for more details, check *weight* parameter for [folium.vector_layers.PolyLine](https://python-visualization.github.io/folium/modules.html#folium.vector_layers.PolyLine)).
- `line_opacity`: *float*, default: *0.6*. Stroke opacity (for more details, check *opacity* parameter for [folium.vector_layers.PolyLine](https://python-visualization.github.io/folium/modules.html#folium.vector_layers.PolyLine)).
- `line_smooth_factor`: *float*, default: *1.0*. How much to simplify the polyline on each zoom level. More means better performance and smoother look, and less means more accurate representation (for more details, check *smooth_factor* parameter for [folium.vector_layers.PolyLine](https://python-visualization.github.io/folium/modules.html#folium.vector_layers.PolyLine)).

<br>

### copy_activities

```.py
copy_activities(activities_files=activities['filename'])
```

#### Description
- Copies a given .fit/.gpx/.tcx list of files to 'output\activities' folder.

#### Parameters
- `activities_files`: *list*, default: *activities['filename']*.



## Save map as a high definition .png file and print it on canvas

Unfortunately Folium does not natively export a rendered map to .png.

A workaround is to open the rendered .html Folium map in Chrome, then open Chrome's Inspector, changing the width and high dimensions to 3500 x 3500 px, setting the zoom to 22% and the DPR to 3.0. Then capture a full size screenshot.

The [canvas.xcf](canvas.xcf) is a Gimp template for printing a canvas in 30 x 30 cm. Its design is similar to [this](https://www.reddit.com/r/bicycling/comments/7hiv41/i_printed_my_strava_heatmap_on_canvas_infos_in/) Reddit discussion:

<p align="center">
<img src="./media/heatmap_munich_2.png" alt="Heatmap Munich" width=800>
</p>

The statistics shown in the lower right corner are printed once the `heatmap` function is executed.


# Documentation

[Strava API v3](https://developers.strava.com/docs/reference/#api-models-DetailedActivity): Definition of activities variables.


# See also

## Similar projects

These repositories have a similar or additional purpose to this project:

[Strava local heatmap browser](https://github.com/remisalmon/Strava-local-heatmap-browser): Code to reproduce the Strava Global Heatmap with local .gpx files (Python).

[Visualization of activities from Garmin Connect](https://github.com/andyakrn/activities_heatmap): Code for processing activities with .gpx files from Garmin Connect (Python).

[Create artistic visualisations with your Strava exercise data](https://github.com/marcusvolz/strava_py): Code for creating artistic visualizations with your Strava exercise data (Python; a R version is available [here](https://github.com/marcusvolz/strava)).

[strava-offline](https://github.com/liskin/strava-offline): Tool to keep a local mirror of Strava activities for further analysis/processing.

[dérive - Generate a heatmap from GPS tracks](https://erik.github.io/derive/): Generate heatmap by drag and dropping one or more .gpx/.tcx/.fit/.igc/.skiz file(s) (JavaScript, HTML).


## Articles

[Data Science For Cycling - How to Visualize GPX Strava Routes With Python and Folium](https://towardsdatascience.com/data-science-for-cycling-how-to-visualize-gpx-strava-routes-with-python-and-folium-21b96ade73c7) ([GitHub](https://github.com/better-data-science/data-science-for-cycling)).

[Build Interactive GPS activity maps from GPX files using Folium](https://towardsdatascience.com/build-interactive-gps-activity-maps-from-gpx-files-using-folium-cf9eebba1fe7) ([GitHub](https://github.com/datachico/gpx_to_folium_maps)).

## External links

[StatsHunters](https://www.statshunters.com): Connect your Strava account and show all your sport activities and added photos on one map.

Recommended [settings](https://www.statshunters.com/settings):
- [x] Receive monthly statistics by email
- [x] Hide my data in club heatmaps

[Cultureplot Custom Strava Heatmap Generator](https://cultureplot.com/strava-heatmap/): Connect to Strava to see your activity heatmap. Includes the possibility to filter the activities (by date, time and type) and to customize the map (map type, background color, line color (also by activity), thickness and opacity).
