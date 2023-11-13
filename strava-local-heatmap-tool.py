## Strava Local Heatmap Tool
# Last update: 2023-10-18


"""About: Create Strava heatmaps locally using Folium library in Python."""


###############
# Initial Setup
###############

# Erase all declared global variables
globals().clear()


# Import packages
from datetime import timedelta
import glob
import gzip
import os
from pathlib import Path
import shutil
import webbrowser

from dateutil import parser

# from fitparse import FitFile
import folium
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
import pandas as pd
from plotnine import aes, geom_line, ggplot, labs, scale_color_brewer, theme_minimal
import sweat


# Set working directory
os.chdir(path=os.path.join(os.path.expanduser('~'), 'Downloads', 'Strava Export'))


###########
# Functions
###########


def gz_extract(*, directory):
    # List of files including path
    files = glob.glob(pathname=os.path.join(directory, '*.gz'), recursive=False)

    if len(files) > 0:
        for file in files:
            # Get file name without extension
            file_name = Path(file).stem

            # Extract file
            with gzip.open(filename=file, mode='rb', encoding=None) as file_in, open(
                os.path.join(os.getcwd(), directory, file_name),
                mode='wb',
                encoding=None,
            ) as file_out:
                shutil.copyfileobj(fsrc=file_in, fdst=file_out)

            # Delete file
            os.remove(path=file)


def tcx_lstrip(*, directory):
    """Remove leading first line blank spaces of .tcx activity files."""
    # List of .tcx files including path
    files = glob.glob(pathname=os.path.join(directory, '*.tcx'), recursive=False)

    if len(files) > 0:
        for file in files:
            with open(file=file, encoding='utf-8') as file_in:
                file_text = file_in.readlines()
                file_text_0 = file_text[0]
                file_text[0] = file_text[0].lstrip()

            if file_text[0] != file_text_0:
                with open(file=file, mode='w', encoding='utf-8') as file_out:
                    file_out.writelines(file_text)


def rename_columns(*, df):
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(pat=r' |\.|-|/', repl=r'_', regex=True)
        .str.replace(pat=r':', repl=r'', regex=True)
        .str.replace(pat=r'__', repl=r'_', regex=True)
    )

    # Return objects
    return df


# def read_fit(*, activities_file):
#     """Import .fit file to DataFrame."""
#
#     # Import .fit file
#     fitfile = FitFile(activities_file)
#
#     total = []
#
#     # Convert .fit file to DataFrame
#     for record in fitfile.get_messages('record'):
#         values = record.get_values()
#         total.append(values)
#
#     df = pd.DataFrame(data=total, index=None, dtype=None)
#
#     # Rename columns
#     df = df.rename(columns={'timestamp': 'datetime', 'position_lat': 'latitude', 'position_long': 'longitude'})
#
#     # Create 'filename' column
#     df['filename'] = activities_file
#
#     # Transform columns
#     if 'latitude' in df.columns:
#         # df['latitude'] = df['latitude'].map(lambda row: (row * 180 / 2**31 + 180) % 360 - 180)
#         df['latitude'] = df['latitude'].map(lambda row: row * 180 / (2 << 30))
#
#     if 'longitude' in df.columns:
#         # df['longitude'] = df['longitude'].map(lambda row: (row * 180 / 2**31 + 180) % 360 - 180)
#         df['longitude'] = df['longitude'].map(lambda row: row * 180 / (2 << 30))
#
#
#     # Select columns
#     df = df.filter(items=['datetime', 'filename', 'latitude', 'longitude'])
#
#     # Return objects
#     return df


def activities_coordinates_import(*, activities_folder):
    """Import .fit/.gpx/.tcx activity files into a DataFrame."""
    # List of .fit/.gpx/.tcx files to be imported
    activities_files = glob.glob(
        pathname=os.path.join(activities_folder, '*.fit'),
        recursive=False,
    )
    activities_files.extend(
        glob.glob(
            pathname=os.path.join(activities_folder, '*.gpx'),
            recursive=False,
        ),
    )
    activities_files.extend(
        glob.glob(
            pathname=os.path.join(activities_folder, '*.tcx'),
            recursive=False,
        ),
    )

    # Create empty DataFrame
    activities_coordinates_df = pd.DataFrame(data=None, index=None, dtype='str')

    # Import activities
    for activities_file in activities_files:
        try:
            # Import file and convert to DataFrame
            df = sweat.read_file(fpath=activities_file)

            # Create 'filename' column
            df['filename'] = activities_file
            df['filename'] = df['filename'].replace(
                to_replace=r'^activities',
                value=r'',
                regex=True,
            )
            df['filename'] = df['filename'].replace(
                to_replace=r'^/[/]?|\\[\\]?',
                value=r'',
                regex=True,
            )

            # Concatenate DataFrame
            activities_coordinates_df = pd.concat(
                objs=[activities_coordinates_df, df],
                axis=0,
                ignore_index=False,
                sort=False,
            )

        except Exception:
            pass

    activities_coordinates_df = activities_coordinates_df.filter(
        items=['datetime', 'filename', 'latitude', 'longitude'],
    )

    # Get elapsed time (in seconds)
    # activities_coordinates_df['elapsed_time'] = activities_coordinates_df.groupby(by=['filename'], level=None, as_index=False, sort=True, dropna=True)['datetime'].transform(lambda row: (row.max() - row.min()).total_seconds())

    # Remove rows without latitude/longitude
    activities_coordinates_df = activities_coordinates_df[
        activities_coordinates_df['latitude'].notna()
    ]

    # Return objects
    return activities_coordinates_df


def activities_geolocator(*, activities_coordinates_df, skip_geolocation=False):
    """Get geolocation for .fit/.gpx/.tcx activity files given the start recorded coordinates (first non-missing latitude/longitude)."""
    # Settings and variables
    geolocator = Nominatim(
        domain='nominatim.openstreetmap.org',
        scheme='https',
        user_agent='strava-local-heatmap-tool',
    )
    reverse = RateLimiter(func=geolocator.reverse, min_delay_seconds=1)

    # Create 'activities_geolocation' DataFrame
    activities_geolocation = (
        activities_coordinates_df
        # Rename columns
        .rename(
            columns={
                'latitude': 'activity_location_latitude',
                'longitude': 'activity_location_longitude',
            },
        )
        # Keep first row of each filename
        .groupby(
            by=['filename'],
            level=None,
            as_index=False,
            sort=True,
            dropna=True,
        ).first()
    )

    if skip_geolocation is False:
        # Create 'activity_geolocation' column
        activities_geolocation['activity_geolocation'] = activities_geolocation.apply(
            lambda row: reverse(
                query='{}, {}'.format(
                    row['activity_location_latitude'],
                    row['activity_location_longitude'],
                ),
                exactly_one=True,
                addressdetails=True,
                namedetails=True,
                language='en',
                timeout=None,
            )
            if pd.notna(row['activity_location_latitude'])
            else None,
            axis=1,
        )

        # Create 'activity_location_country_code' column
        activities_geolocation[
            'activity_location_country_code'
        ] = activities_geolocation.apply(
            lambda row: row['activity_geolocation']
            .raw.get('address')
            .get('country_code')
            if pd.notna(row['activity_geolocation'])
            else None,
            axis=1,
        )

        # Create 'activity_location_country' column
        activities_geolocation[
            'activity_location_country'
        ] = activities_geolocation.apply(
            lambda row: row['activity_geolocation'].raw.get('address').get('country')
            if pd.notna(row['activity_geolocation'])
            else None,
            axis=1,
        )

        # Create 'activity_location_state' column
        activities_geolocation[
            'activity_location_state'
        ] = activities_geolocation.apply(
            lambda row: row['activity_geolocation'].raw.get('address').get('state')
            if pd.notna(row['activity_geolocation'])
            else None,
            axis=1,
        )

        # Create 'activity_location_city' column
        activities_geolocation['activity_location_city'] = activities_geolocation.apply(
            lambda row: row['activity_geolocation'].raw.get('address').get('city')
            if pd.notna(row['activity_geolocation'])
            else None,
            axis=1,
        )

        # Create 'activity_location_postal_code' column
        activities_geolocation[
            'activity_location_postal_code'
        ] = activities_geolocation.apply(
            lambda row: row['activity_geolocation'].raw.get('address').get('postcode')
            if pd.notna(row['activity_geolocation'])
            else None,
            axis=1,
        )

        activities_geolocation = (
            activities_geolocation
            # Remove columns
            .drop(columns=['datetime', 'activity_geolocation'], axis=1, errors='ignore')
        )

    if skip_geolocation is True:
        activities_geolocation = activities_geolocation.assign(
            activity_location_country_code=None,
            activity_location_country=None,
            activity_location_state=None,
            activity_location_city=None,
            activity_location_postal_code=None,
            activity_location_latitude=None,
            activity_location_longitude=None,
        )

    activities_geolocation = (
        activities_geolocation
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
    return activities_geolocation


def activities_import(*, activities_folder, activities_file, skip_geolocation=False):
    """
    Strava activities import.

    Strava's activities column definitions - https://developers.strava.com/docs/reference/#api-models-DetailedActivity
    elapsed_time, moving_time: seconds
    distance, elevation_gain, elevation_loss: meters
    max_speed, average_speed: meters/second
    """
    # Import .fit/.gpx/.tcx activity files into a DataFrame
    activities_coordinates_df = activities_coordinates_import(
        activities_folder=activities_folder,
    )

    # Get geolocation for .fit/.gpx/.tcx activity files given the start recorded coordinates (first non-missing latitude/longitude)
    activities_geolocation = activities_geolocator(
        activities_coordinates_df=activities_coordinates_df,
        skip_geolocation=skip_geolocation,
    )

    # Import Strava activities
    activities_df = pd.read_csv(
        filepath_or_buffer=activities_file,
        sep=',',
        header=0,
        index_col=None,
        skiprows=0,
        skipfooter=0,
        dtype=None,
        engine='python',
        encoding='utf-8',
        keep_default_na=True,
    )

    # Rename columns
    activities_df = rename_columns(df=activities_df)

    activities_df = (
        activities_df
        # Clean 'filename' column
        .assign(
            filename=lambda row: row['filename'].replace(
                to_replace=r'^activities/|\.gz$',
                value='',
                regex=True,
            ),
        )
        # Left join 'activities_geolocation'
        .merge(
            right=activities_geolocation,
            how='left',
            on=['filename'],
            indicator=False,
        )
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
        .astype(dtype={'activity_id': 'str'}).assign(
            activity_date=lambda row: row['activity_date'].apply(parser.parse),
        )
        # Transform columns
        .assign(
            elapsed_time=lambda row: row['elapsed_time'] / 60,
            moving_time=lambda row: row['moving_time'] / 60,
            max_speed=lambda row: row['max_speed'] * 3.6,
            average_speed=lambda row: row['average_speed'] * 3.6,
        )
        # Rearrange rows
        .sort_values(by=['activity_date', 'activity_type'], ignore_index=True)
    )

    # Return objects
    return activities_df


def activities_filter(
    *,
    activities_df,
    activity_type=None,
    activity_location_state=None,
    bounding_box=None,
):
    """Filter Strava activities DataFrame."""
    # Filter activities by type
    if activity_type is not None:
        activities_df = activities_df.query('activity_type.isin(@activity_type)')

    # Filter activities by state
    if activity_location_state is not None:
        activities_df = activities_df.query(
            'activity_location_state.isin(@activity_location_state)',
        ).reset_index(level=None, drop=True)

    # Filter activities inside a bounding box
    if all(value is not None for value in bounding_box.values()):
        activities_df = activities_df[
            activities_df['activity_location_latitude'].between(
                min(
                    bounding_box['latitude_bottom_left'],
                    bounding_box['latitude_bottom_right'],
                ),
                max(
                    bounding_box['latitude_top_left'],
                    bounding_box['latitude_top_right'],
                ),
            )
        ]
        activities_df = activities_df[
            activities_df['activity_location_longitude'].between(
                min(
                    bounding_box['longitude_bottom_left'],
                    bounding_box['longitude_top_left'],
                ),
                max(
                    bounding_box['longitude_bottom_right'],
                    bounding_box['longitude_top_right'],
                ),
            )
        ]

    # Return objects
    return activities_df


def heatmap(
    *,
    activities_df,
    activities_coordinates_df,
    activity_colors=None,
    map_tile='dark_all',
    map_zoom_start=12,
    line_weight=1.0,
    line_opacity=0.6,
    line_smooth_factor=1.0,
):
    """Create Heatmap based on inputted activities DataFrame."""
    activities_df = (
        activities_df
        # Remove activities without latitude/longitude coordinates
        .query('filename.notna()')
    )

    if 'distance' not in activities_df.columns:
        activities_df = activities_df.assign(distance=0)

    if 'datetime' not in activities_coordinates_df.columns:
        activities_coordinates_df = activities_coordinates_df.assign(
            datetime=pd.Timestamp.now(tz='UTC').replace(tzinfo=None),
        )

    activities_coordinates_df = (
        activities_coordinates_df
        # Filter activities coordinates given the filtered activities
        .query('filename.isin(@activities_df["filename"])')
        # Select columns
        .filter(items=['datetime', 'filename', 'latitude', 'longitude'])
        # Left join 'activities_df'
        .merge(
            right=activities_df.filter(
                items=['filename', 'activity_id', 'activity_type', 'distance'],
            ),
            how='left',
            on=['filename'],
            indicator=False,
        )
        # Remove columns
        .drop(columns=['filename'], axis=1, errors='ignore')
    )

    # Test memory usage
    # activities_coordinates_df.info(memory_usage='deep')

    # Transform columns
    activities_coordinates_df['coordinates'] = list(
        zip(
            activities_coordinates_df['latitude'],
            activities_coordinates_df['longitude'],
        ),
    )

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
        df_activity_type = activities_coordinates_df[
            activities_coordinates_df['activity_type'] == activity_type
        ]

        for activity in df_activity_type['activity_id'].unique():
            date = df_activity_type[df_activity_type['activity_id'] == activity][
                'datetime'
            ].dt.date.iloc[0]
            distance = round(
                df_activity_type[df_activity_type['activity_id'] == activity][
                    'distance'
                ].iloc[0]
                / 1000,
                1,
            )

            coordinates = tuple(
                df_activity_type[df_activity_type['activity_id'] == activity][
                    'coordinates'
                ],
            )
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

    # Create 'output' folder
    os.makedirs(name='output', exist_ok=True)

    # Save to .html file
    activities_map.save(outfile=os.path.join('output', 'activities-map.html'))
    webbrowser.open(url=os.path.join('output', 'activities-map.html'))

    # Summary
    print('Total activities: ' + str(activities_df['activity_id'].nunique()))
    print(
        'Total distance (in km): '
        + str(round(activities_df['distance'].sum() / 1000, 1)),
    )
    print(
        'Total moving time (in days, hours, minutes, seconds): '
        + str(
            timedelta(
                seconds=(
                    activities_df.assign(moving_time=activities_df['moving_time'] * 60)[
                        'moving_time'
                    ]
                ).sum(),
            ),
        ),
    )
    print(
        'Total elevation gain (in km): '
        + str(round(activities_df['elevation_gain'].sum() / 1000, 1)),
    )
    print(
        'Longest activity (in km): '
        + round(
            activities_df[
                activities_df['distance'] == activities_df['distance'].max()
            ].filter(
                items=['distance'],
            )
            / 1000,
            1,
        ).to_string(index=False, header=False)
        + ' ('
        + activities_df[activities_df['distance'] == activities_df['distance'].max()]
        .filter(items=['activity_date'])
        .assign(activity_date=lambda row: row['activity_date'].dt.strftime('%b %Y'))
        .to_string(index=False, header=False)
        + ')',
    )
    print('Max speed (km/h): ' + str(round(activities_df['max_speed'].max(), 1)))
    print(
        'Average speed (km/h): ' + str(round(activities_df['average_speed'].mean(), 1)),
    )


# Copy activities files to 'output/activities' folder
def copy_activities(*, activities_files):
    # Create 'output/activities' folder
    os.makedirs(name=os.path.join('output', 'activities'), exist_ok=True)

    # Copy activities files to 'output/activities' folder
    for filename in activities_files.tolist():
        shutil.copy(
            src=os.path.join('activities', filename),
            dst=os.path.join('output', 'activities'),
        )


# Rename activities files
def activities_file_rename(
    *,
    activities_geolocation_df,
    activities_folder='activities',
):
    # List of .fit/.gpx/.tcx files to be renamed
    activities_files = glob.glob(
        pathname=os.path.join(activities_folder, '*.fit'),
        recursive=False,
    )
    activities_files.extend(
        glob.glob(
            pathname=os.path.join(activities_folder, '*.gpx'),
            recursive=False,
        ),
    )
    activities_files.extend(
        glob.glob(
            pathname=os.path.join(activities_folder, '*.tcx'),
            recursive=False,
        ),
    )

    # New file name
    activities_geolocation_df['reference'] = activities_geolocation_df[
        [
            'activity_location_country',
            'activity_location_state',
            'activity_location_city',
            'filename',
        ]
    ].apply(lambda row: '-'.join(column for column in row if pd.notna(column)), axis=1)
    activities_geolocation_df['reference'] = activities_geolocation_df[
        'reference'
    ].replace(to_replace=r'/', value=r'-', regex=True)

    references = dict(
        activities_geolocation_df.dropna(subset=['filename']).set_index(
            keys='filename',
            drop=True,
            append=False,
        )['reference'],
    )

    for activity_file in activities_files:
        activity_file = Path(activity_file)

        filename_new = references.get(activity_file.name, activity_file.stem)
        activity_file.rename(
            target=activity_file.with_name(f'{filename_new}{activity_file.suffix}'),
        )


###########################
# Strava Local Heatmap Tool
###########################

# Extract .gz files
gz_extract(directory='activities')

# Remove leading first line blank spaces of .tcx activity files
tcx_lstrip(directory='activities')

# Import Strava activities to DataFrame
activities_df = activities_import(
    activities_folder='activities',
    activities_file='activities.csv',
    skip_geolocation=False,
)


## Tests

# Check for activities without activity_gear
(
    activities_df.query('activity_gear.isna()')
    .groupby(
        by=['activity_type'],
        level=None,
        as_index=False,
        sort=True,
        dropna=True,
    )
    .agg(count=('activity_id', 'nunique'))
)


# Check for activity_name inconsistencies
(activities_df.query('activity_name.str.contains(r"^ |  | $")'))

(activities_df.query('activity_name.str.contains(r"[^\\s]-|-[^\\s]")'))


# Check for distinct values for activity_name separated by a hyphen
(
    pd.DataFrame(
        data=(
            activities_df.query('activity_type == "Ride"')['activity_name']
            .str.split(pat=' - ', expand=True)
            .stack()
            .unique()
        ),
        index=None,
        columns=['activity_name'],
        dtype=None,
    ).sort_values(by=['activity_name'], ignore_index=True)
)


# Check for distinct values for activity_description
(
    pd.DataFrame(
        data=(
            activities_df.query(
                'activity_type == "Weight Training" and activity_description.notna()',
            )['activity_description']
            .replace(to_replace=r'; | and ', value=r', ', regex=True)
            .str.lower()
            .str.split(pat=',', expand=True)
            .stack()
            .unique()
        ),
        index=None,
        columns=['activity_description'],
        dtype=None,
    ).sort_values(by=['activity_description'], ignore_index=True)
)


## Summary

# Count of activities by type
(
    activities_df.groupby(
        by=['activity_type'],
        level=None,
        as_index=False,
        sort=True,
        dropna=True,
    ).agg(count=('activity_id', 'nunique'))
)


# Runs overview per year-month (distance in km)
(
    activities_df.query('activity_type == "Run"')
    .assign(activity_month=lambda row: row['activity_date'].dt.strftime('%Y-%m'))
    .groupby(
        by=['activity_month'],
        level=None,
        as_index=False,
        sort=True,
        dropna=True,
    )
    .agg(
        count=('activity_id', 'nunique'),
        distance=('distance', lambda x: x.sum() / 1000),
    )
)


# Strava yearly overview cumulative (Plot)
strava_yearly_overview = (
    activities_df.query('activity_type == "Ride"')
    .query('activity_date >= "2017-01-01" and activity_date < "2023-01-01"')
    .assign(
        distance=lambda row: row['distance'] / 1000,
        year=lambda row: row['activity_date'].dt.strftime('%Y'),
        day_of_year=lambda row: row['activity_date'].dt.dayofyear,
    )
    .assign(
        distance_cumulative=lambda row: row.groupby(
            by=['year'],
            level=None,
            as_index=False,
            sort=True,
            dropna=True,
        )['distance'].transform('cumsum'),
    )
    .filter(
        items=[
            'activity_date',
            'year',
            'day_of_year',
            'distance',
            'distance_cumulative',
        ],
    )
)

(
    ggplot(
        strava_yearly_overview,
        aes(
            x='day_of_year',
            y='distance_cumulative',
            group='year',
            color='factor(year)',
        ),
    )
    + geom_line()
    + scale_color_brewer(palette=1)
    + theme_minimal()
    +
    # theme(legend_position='bottom') +
    labs(
        title='Cumultative Distance (KM)',
        y='Distance (KM)',
        x='Day of Year',
        color='Year',
    )
)

# Delete objects
del strava_yearly_overview


# Filter Strava activities
activities_df = activities_filter(
    activities_df=activities_df,
    activity_type=['Hike', 'Ride', 'Run'],
    activity_location_state=None,
    bounding_box={
        'latitude_top_right': None,
        'longitude_top_right': None,
        'latitude_top_left': None,
        'longitude_top_left': None,
        'latitude_bottom_left': None,
        'longitude_bottom_left': None,
        'latitude_bottom_right': None,
        'longitude_bottom_right': None,
    },
)


# Create heatmap
heatmap(
    activities_df=activities_df,
    activities_coordinates_df=activities_coordinates_df,
    activity_colors={'Hike': '#FF0000', 'Ride': '#00A3E0', 'Run': '#FF0000'},
    map_tile='dark_all',
    map_zoom_start=12,
    line_weight=1.0,
    line_opacity=0.6,
    line_smooth_factor=1.0,
)


# Copy activities files to 'output/activities' folder
# copy_activities(activities_files=activities_df['filename'])


# Import .fit/.gpx/.tcx activity files into a DataFrame
# activities_coordinates_df = activities_coordinates_import(activities_folder='activities')


# Get geolocation for .fit/.gpx/.tcx activity files given the start recorded coordinates (first non-missing latitude/longitude)
# activities_geolocation = activities_geolocator(activities_coordinates_df=activities_coordinates_df, skip_geolocation=False)


# activities_file_rename(activities_geolocation_df=activities_geolocation, activities_folder='activities')
