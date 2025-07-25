"""About: Create Strava heatmaps locally using Folium library in Python."""

# Import packages

from datetime import timedelta
import glob
import gzip
import os
from pathlib import Path
import shutil
import webbrowser

from dateutil import parser
from fitparse import FitFile
import folium
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
import gpxpy
import gpxpy.gpx
from janitor import clean_names
import pandas as pd
from pandas import DataFrame
from tcxreader.tcxreader import TCXReader


# Settings

## Copy-on-Write (will be enabled by default in version 3.0)
if pd.__version__ >= '1.5.0' and pd.__version__ < '3.0.0':
    pd.options.mode.copy_on_write = True


# Functions


def gz_extract(*, activities_directory: str) -> None:
    # List of files including path
    files = glob.glob(pathname=os.path.join(activities_directory, '*.gz'), recursive=False)

    if len(files) > 0:
        for file in files:
            # Get file name without extension
            file_name = Path(file).stem

            # Extract file
            with gzip.open(filename=file, mode='rb', encoding=None) as file_in, open(os.path.join(os.getcwd(), activities_directory, file_name), mode='wb', encoding=None) as file_out:
                shutil.copyfileobj(fsrc=file_in, fdst=file_out)

            # Delete file
            os.remove(path=file)


def tcx_lstrip(*, activities_directory: str) -> None:
    """Remove leading first line blank spaces of .tcx activity files."""
    # List of .tcx files including path
    files = glob.glob(pathname=os.path.join(activities_directory, '*.tcx'), recursive=False)

    if len(files) > 0:
        for file in files:
            with open(file=file, encoding='utf-8') as file_in:
                file_text = file_in.readlines()
                file_text_0 = file_text[0]
                file_text[0] = file_text[0].lstrip()

            if file_text[0] != file_text_0:
                with open(file=file, mode='w', encoding='utf-8') as file_out:
                    file_out.writelines(file_text)


def activity_file_parse(*, file_path: str) -> DataFrame:
    parsed_data = []

    if file_path.endswith('.fit'):
        fitfile = FitFile(file_path)
        for record in fitfile.get_messages('record'):
            data = {field.name: field.value for field in record}
            if 'timestamp' in data and 'position_lat' in data and 'position_long' in data:
                parsed_data.append(
                    {
                        'datetime': data.get('timestamp'),
                        'latitude': data.get('position_lat') * (180 / 2**31),  # Convert to degrees
                        'longitude': data.get('position_long') * (180 / 2**31),  # Convert to degrees
                    },
                )

    elif file_path.endswith('.gpx'):
        with open(file_path) as gpx_file:
            gpx = gpxpy.parse(gpx_file)
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        if point.time is not None and point.latitude is not None and point.longitude is not None:
                            parsed_data.append(
                                {
                                    'datetime': point.time,
                                    'latitude': point.latitude,
                                    'longitude': point.longitude,
                                },
                            )

    elif file_path.endswith('.tcx'):
        tcx_reader = TCXReader()
        tcx = tcx_reader.read(file_path)
        for trackpoint in tcx.trackpoints:
            if trackpoint.time is not None and trackpoint.latitude is not None and trackpoint.longitude is not None:
                parsed_data.append(
                    {
                        'datetime': trackpoint.time,
                        'latitude': trackpoint.latitude,
                        'longitude': trackpoint.longitude,
                    },
                )

    # Create DataFrame (empty if "parsed_data" is empty)
    df = pd.DataFrame(data=parsed_data, index=None, dtype=None)

    # Remove timezone information for .gpx files
    if file_path.endswith('.gpx') and not df.empty:
        df = df.assign(datetime=df['datetime'].dt.tz_localize(tz=None))

    # Return objects
    return df


def activities_coordinates_import(*, activities_directory: str) -> DataFrame:
    """Import .fit/.gpx/.tcx activity files into a DataFrame."""
    # List of .fit/.gpx/.tcx files to be imported
    activities_files = glob.glob(pathname=os.path.join(activities_directory, '*.fit'), recursive=False)
    activities_files.extend(glob.glob(pathname=os.path.join(activities_directory, '*.gpx'), recursive=False))
    activities_files.extend(glob.glob(pathname=os.path.join(activities_directory, '*.tcx'), recursive=False))

    # Create empty DataFrame
    activities_coordinates_df = pd.DataFrame(data=None, index=None, dtype='str')

    # Import activities
    for activities_file in activities_files:
        try:
            # Import file and convert to DataFrame
            df = activity_file_parse(file_path=activities_file)

            # Create 'filename' column
            df['filename'] = activities_file
            df['filename'] = df['filename'].replace(to_replace=r'.*activities', value=r'', regex=True)
            df['filename'] = df['filename'].replace(to_replace=r'^/[/]?|\\[\\]?', value=r'', regex=True)

            # Concatenate DataFrame
            activities_coordinates_df = pd.concat(objs=[activities_coordinates_df, df], axis=0, ignore_index=False, sort=False)

        except Exception:
            pass

    activities_coordinates_df = activities_coordinates_df.filter(items=['datetime', 'filename', 'latitude', 'longitude'])

    # Get elapsed time (in seconds)
    # activities_coordinates_df['elapsed_time'] = activities_coordinates_df.groupby(by=['filename'], level=None, as_index=False, sort=True, dropna=True)['datetime'].transform(lambda row: (row.max() - row.min()).total_seconds())

    # Remove rows without latitude/longitude
    if not activities_coordinates_df.empty and 'latitude' in activities_coordinates_df.columns:
        activities_coordinates_df = activities_coordinates_df[activities_coordinates_df['latitude'].notna()]
    else:
        print('No activities with GPS data (latitude/longitude) found.')

    # Return objects
    return activities_coordinates_df


def activities_geolocator(*, activities_coordinates_df: DataFrame, skip_geolocation: bool = True) -> DataFrame:
    """Get geolocation for .fit/.gpx/.tcx activity files given the start recorded coordinates (first non-missing latitude/longitude)."""
    # Settings and variables
    geolocator = Nominatim(domain='nominatim.openstreetmap.org', scheme='https', user_agent='strava-local-heatmap-tool')
    reverse = RateLimiter(func=geolocator.reverse, min_delay_seconds=1)

    # Create 'activities_geolocation_df' DataFrame
    activities_geolocation_df = (
        activities_coordinates_df
        # Rename columns
        .rename(
            columns={
                'latitude': 'activity_location_latitude',
                'longitude': 'activity_location_longitude',
            },
        )
        # Keep first row of each filename
        .groupby(by=['filename'], level=None, as_index=False, sort=True, dropna=True)
        .first()
    )

    if skip_geolocation is False:
        # Create 'activity_geolocation' column
        activities_geolocation_df['activity_geolocation'] = activities_geolocation_df.apply(
            lambda row: (
                reverse(query='{}, {}'.format(row['activity_location_latitude'], row['activity_location_longitude']), exactly_one=True, addressdetails=True, namedetails=True, language='en', timeout=None)
                if pd.notna(row['activity_location_latitude'])
                else None
            ),
            axis=1,
        )

        # Create 'activity_location_country_code' column
        activities_geolocation_df['activity_location_country_code'] = activities_geolocation_df.apply(
            lambda row: (row['activity_geolocation'].raw.get('address').get('country_code') if pd.notna(row['activity_geolocation']) else None),
            axis=1,
        )

        # Create 'activity_location_country' column
        activities_geolocation_df['activity_location_country'] = activities_geolocation_df.apply(
            lambda row: (row['activity_geolocation'].raw.get('address').get('country') if pd.notna(row['activity_geolocation']) else None),
            axis=1,
        )

        # Create 'activity_location_state' column
        activities_geolocation_df['activity_location_state'] = activities_geolocation_df.apply(
            lambda row: (row['activity_geolocation'].raw.get('address').get('state') if pd.notna(row['activity_geolocation']) else None),
            axis=1,
        )

        # Create 'activity_location_city' column
        activities_geolocation_df['activity_location_city'] = activities_geolocation_df.apply(
            lambda row: (row['activity_geolocation'].raw.get('address').get('city') if pd.notna(row['activity_geolocation']) else None),
            axis=1,
        )

        # Create 'activity_location_postal_code' column
        activities_geolocation_df['activity_location_postal_code'] = activities_geolocation_df.apply(
            lambda row: (row['activity_geolocation'].raw.get('address').get('postcode') if pd.notna(row['activity_geolocation']) else None),
            axis=1,
        )

        activities_geolocation_df = (
            activities_geolocation_df
            # Remove columns
            .drop(columns=['datetime', 'activity_geolocation'], axis=1, errors='ignore')
        )

    if skip_geolocation is True:
        activities_geolocation_df = activities_geolocation_df.assign(
            activity_location_country_code=None,
            activity_location_country=None,
            activity_location_state=None,
            activity_location_city=None,
            activity_location_postal_code=None,
            activity_location_latitude=None,
            activity_location_longitude=None,
        )

    activities_geolocation_df = (
        activities_geolocation_df
        # Select columns
        .filter(
            items=[
                'filename',
                'activity_location_country_code',
                'activity_location_country',
                'activity_location_state',
                'activity_location_city',
                'activity_location_postal_code',
                'activity_location_latitude',
                'activity_location_longitude',
            ],
        )
        # Rearrange rows
        .sort_values(by=['filename'], ignore_index=True)
    )

    # Return objects
    return activities_geolocation_df


def activities_import(*, activities_directory: str, activities_file: str, skip_geolocation: bool = True) -> tuple[DataFrame, DataFrame]:
    """
    Strava activities import.

    Strava's activities column definitions - https://developers.strava.com/docs/reference/#api-models-DetailedActivity
    elapsed_time, moving_time: seconds
    distance, elevation_gain, elevation_loss: meters
    max_speed, average_speed: meters/second
    """
    # Import .fit/.gpx/.tcx activity files into a DataFrame
    activities_coordinates_df = activities_coordinates_import(activities_directory=activities_directory)

    # Get geolocation for .fit/.gpx/.tcx activity files given the start recorded coordinates (first non-missing latitude/longitude)
    activities_geolocation_df = activities_geolocator(activities_coordinates_df=activities_coordinates_df, skip_geolocation=skip_geolocation)

    # Import Strava activities
    activities_df = pd.read_csv(filepath_or_buffer=activities_file, sep=',', header=0, index_col=None, skiprows=0, skipfooter=0, dtype=None, engine='python', encoding='utf-8', keep_default_na=True)

    # Rename columns
    activities_df = clean_names(activities_df)

    activities_df = (
        activities_df
        # Clean 'filename' column
        .assign(filename=lambda row: row['filename'].replace(to_replace=r'^activities/|\.gz$', value='', regex=True))
        # Left join 'activities_geolocation_df'
        .merge(right=activities_geolocation_df, how='left', on=['filename'], indicator=False)
        # Remove columns
        .drop(columns=['distance', 'commute'], axis=1, errors='ignore')
        # Remame columns
        .rename(
            columns={
                'distance_1': 'distance',
                'commute_1': 'commute',
                '<span_class="translation_missing"_title="translation_missing_en_us_lib_export_portability_exporter_activities_horton_values_total_steps">total_steps<_span>': 'steps',
            },
        )
        # Select columns
        .filter(
            items=[
                'activity_date',
                'activity_type',
                'activity_id',
                'activity_name',
                'activity_description',
                'filename',
                'from_upload',
                'activity_location_country_code',
                'activity_location_country',
                'activity_location_state',
                'activity_location_city',
                'activity_location_postal_code',
                'activity_location_latitude',
                'activity_location_longitude',
                'commute',
                'activity_gear',
                'elapsed_time',
                'moving_time',
                'distance',
                'max_speed',
                'average_speed',
                'steps',
                'elevation_gain',
                'elevation_loss',
                'elevation_low',
                'elevation_high',
                'max_grade',
                'average_grade',
                'grade_adjusted_distance',
                'max_heart_rate',
                'average_heart_rate',
                'max_cadence',
                'average_cadence',
                'max_watts',
                'average_watts',
                'calories',
                'relative_effort',
                'weighted_average_power',
                'power_count',
                'perceived_exertion',
                'perceived_relative_effort',
                'total_weight_lifted',
                'athlete_weight',
                'bike_weight',
                'max_temperature',
                'average_temperature',
            ],
        )
        # Change dtypes
        .astype(dtype={'activity_id': 'str'})
        .assign(activity_date=lambda row: row['activity_date'].apply(parser.parse))
        # Transform columns
        .assign(elapsed_time=lambda row: row['elapsed_time'] / 60, moving_time=lambda row: row['moving_time'] / 60, max_speed=lambda row: row['max_speed'] * 3.6, average_speed=lambda row: row['average_speed'] * 3.6)
        # Rearrange rows
        .sort_values(by=['activity_date', 'activity_type'], ignore_index=True)
    )

    # Return objects
    return activities_df, activities_coordinates_df


def activities_filter(*, activities_df: DataFrame, activity_type: list[str] | None = None, activity_location_state: list[str] | None = None, bounding_box: dict[str, float] | None = None) -> DataFrame:
    """Filter Strava activities DataFrame."""
    # Filter activities by type
    if activity_type is not None:
        activities_df = activities_df.query(expr='activity_type.isin(@activity_type)')

    # Filter activities by state
    if activity_location_state is not None:
        activities_df = activities_df.query(expr='activity_location_state.isin(@activity_location_state)').reset_index(level=None, drop=True, names=None)

    # Filter activities inside a bounding box
    if all(value is not None for value in bounding_box.values()):
        activities_df = activities_df[
            activities_df['activity_location_latitude'].between(
                min(bounding_box['latitude_bottom_left'], bounding_box['latitude_bottom_right']),
                max(bounding_box['latitude_top_left'], bounding_box['latitude_top_right']),
            )
        ]
        activities_df = activities_df[
            activities_df['activity_location_longitude'].between(
                min(bounding_box['longitude_bottom_left'], bounding_box['longitude_top_left']),
                max(bounding_box['longitude_bottom_right'], bounding_box['longitude_top_right']),
            )
        ]

    # Return objects
    return activities_df


def strava_activities_heatmap(
    *,
    activities_df: DataFrame,
    activities_coordinates_df: DataFrame,
    strava_activities_heatmap_output_path: str,
    activity_colors: dict[str, str] | None = None,
    map_tile: str = 'dark_all',
    map_zoom_start: int = 12,
    line_weight: float = 1.0,
    line_opacity: float = 0.6,
    line_smooth_factor: float = 1.0,
) -> None:
    """Create Heatmap based on inputted activities DataFrame."""
    activities_df = (
        activities_df
        # Remove activities without latitude/longitude coordinates
        .query(expr='filename.notna()')
    )

    if 'distance' not in activities_df.columns:
        activities_df = activities_df.assign(distance=0)

    if 'datetime' not in activities_coordinates_df.columns:
        activities_coordinates_df = activities_coordinates_df.assign(datetime=pd.Timestamp.now(tz='UTC').replace(tzinfo=None))

    activities_coordinates_df = (
        activities_coordinates_df
        # Filter activities coordinates given the filtered activities
        .query(expr='filename.isin(@activities_df["filename"])')
        # Select columns
        .filter(items=['datetime', 'filename', 'latitude', 'longitude'])
        # Left join 'activities_df'
        .merge(right=activities_df.filter(items=['filename', 'activity_id', 'activity_type', 'distance']), how='left', on=['filename'], indicator=False)
        # Remove columns
        .drop(columns=['filename'], axis=1, errors='ignore')
    )

    # Test memory usage
    # activities_coordinates_df.info(memory_usage='deep')

    # Transform columns
    activities_coordinates_df['coordinates'] = list(zip(activities_coordinates_df['latitude'], activities_coordinates_df['longitude']))

    # Define map tile
    if map_tile in ['dark_all', 'dark_nolabels', 'light_all', 'light_nolabels']:
        map_tile = 'https://a.basemaps.cartocdn.com/' + map_tile + '/{z}/{x}/{y}@2x.png'

    if map_tile == 'terrain_background':
        map_tile = 'http://tile.stamen.com/terrain-background/{z}/{x}/{y}.png'

    if map_tile == 'toner_lite':
        map_tile = 'http://tile.stamen.com/toner-lite/{z}/{x}/{y}.png'

    if map_tile == 'ocean_basemap':
        map_tile = 'https://server.arcgisonline.com/ArcGIS/rest/services/Ocean_Basemap/MapServer/tile/{z}/{y}/{x}'

    # Create Folium map
    activities_map = folium.Map(
        tiles=map_tile,
        attr='tile',
        location=[
            round(activities_coordinates_df['latitude'].median(), 4),
            round(activities_coordinates_df['longitude'].median(), 4),
        ],
        zoom_start=map_zoom_start,
    )
    folium.LayerControl().add_to(activities_map)

    # Plot activities into Folium map (adapted from: https://github.com/andyakrn/activities_heatmap)
    for activity_type in activities_coordinates_df['activity_type'].unique():
        df_activity_type = activities_coordinates_df[activities_coordinates_df['activity_type'] == activity_type]

        for activity in df_activity_type['activity_id'].unique():
            date = df_activity_type[df_activity_type['activity_id'] == activity]['datetime'].dt.date.iloc[0]
            distance = round(df_activity_type[df_activity_type['activity_id'] == activity]['distance'].iloc[0] / 1000, 1)

            coordinates = tuple(df_activity_type[df_activity_type['activity_id'] == activity]['coordinates'])
            folium.PolyLine(
                locations=coordinates,
                color=activity_colors[activity_type],
                weight=line_weight,
                opacity=line_opacity,
                control=True,
                name=activity_type,
                popup=folium.Popup(
                    html=(
                        'Activity type: '
                        + activity_type
                        + '<br>'
                        + 'Date: '
                        + str(date)
                        + '<br>'
                        + 'Distance: '
                        + str(distance)
                        + ' km'
                        + '<br>'
                        + '<br>'
                        + '<a href=https://www.strava.com/activities/'
                        + str(activity)
                        + '>'
                        + 'Open in Strava'
                        + '</a>'
                    ),
                    min_width=100,
                    max_width=100,
                ),
                tooltip=activity_type,
                smooth_factor=line_smooth_factor,
                overlay=True,
            ).add_to(activities_map)

    # Save to .html file
    activities_map.save(outfile=strava_activities_heatmap_output_path)
    webbrowser.open(url=strava_activities_heatmap_output_path)

    # Summary
    print('Total activities: ' + str(activities_df['activity_id'].nunique()))
    print('Total distance (in km): ' + str(round(activities_df['distance'].sum() / 1000, 1)))
    print('Total moving time (in days, hours, minutes, seconds): ' + str(timedelta(seconds=(activities_df.assign(moving_time=activities_df['moving_time'] * 60)['moving_time']).sum())))
    print('Total elevation gain (in km): ' + str(round(activities_df['elevation_gain'].sum() / 1000, 1)))
    print(
        'Longest activity (in km): '
        + round(activities_df[activities_df['distance'] == activities_df['distance'].max()].filter(items=['distance']) / 1000, 1).to_string(index=False, header=False)
        + ' ('
        + activities_df[activities_df['distance'] == activities_df['distance'].max()]
        .filter(items=['activity_date'])
        .assign(activity_date=lambda row: row['activity_date'].dt.strftime(date_format='%b %Y'))
        .to_string(index=False, header=False)
        + ')',
    )
    print('Max speed (km/h): ' + str(round(activities_df['max_speed'].max(), 1)))
    print('Average speed (km/h): ' + str(round(activities_df['average_speed'].mean(), 1)))


# Copy activities files to 'output/activities' folder
def copy_activities(*, activities_directory: str, activities_files: pd.Series) -> None:
    # Create 'output/activities' folder
    os.makedirs(name=os.path.join(activities_directory, 'output', 'activities'), exist_ok=True)

    # Copy activities files to 'output/activities' folder
    for filename in activities_files.tolist():
        shutil.copy(src=os.path.join('activities', filename), dst=os.path.join('output', 'activities'))


# Rename activities files
def activities_file_rename(*, activities_directory: str, activities_geolocation_df: DataFrame) -> None:
    # List of .fit/.gpx/.tcx files to be renamed
    activities_files = glob.glob(pathname=os.path.join(activities_directory, '*.fit'), recursive=False)
    activities_files.extend(glob.glob(pathname=os.path.join(activities_directory, '*.gpx'), recursive=False))
    activities_files.extend(glob.glob(pathname=os.path.join(activities_directory, '*.tcx'), recursive=False))

    # New file name
    activities_geolocation_df['reference'] = activities_geolocation_df[
        [
            'activity_location_country',
            'activity_location_state',
            'activity_location_city',
            'filename',
        ]
    ].apply(lambda row: '-'.join(column for column in row if pd.notna(column)), axis=1)
    activities_geolocation_df['reference'] = activities_geolocation_df['reference'].replace(to_replace=r'/', value=r'-', regex=True)

    references = dict(activities_geolocation_df.dropna(subset=['filename']).set_index(keys='filename', drop=True, append=False)['reference'])

    for activity_file in activities_files:
        activity_file = Path(activity_file)

        filename_new = references.get(activity_file.name, activity_file.stem)
        activity_file.rename(target=activity_file.with_name(f'{filename_new}{activity_file.suffix}'))
