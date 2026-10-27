import numpy as np

REGION_VERSION = '2026-10-27_SEP110W_NEP142W_WPWP-original'
REGIONS = {'SEP': {'lat_range': (-35.0, -5.0), 'west': 250.0, 'east_at_south': 290.0, 'east_at_north': 280.0}, 'NEP': {'lat_range': (15.0, 38.0), 'west': 218.0, 'east_at_south': 255.0, 'east_at_north': 235.0}, 'WPWP': {'lat_range': (-10.0, 10.0), 'west': 80.0, 'east_at_south': 150.0, 'east_at_north': 150.0}}

def region_vertices(name):
    region = REGIONS[name]
    south, north = region["lat_range"]
    west = region["west"]
    return np.array([
        [west, south], [region["east_at_south"], south],
        [region["east_at_north"], north], [west, north],
    ])

def make_region_masks(lat, lon):
    latitude = np.asarray(lat)[:, None]
    longitude = np.asarray(lon)[None, :] % 360.
    masks = {}
    for name, region in REGIONS.items():
        south, north = region["lat_range"]
        east = region["east_at_south"] + (
            region["east_at_north"] - region["east_at_south"]
        ) * (latitude - south) / (north - south)
        masks[name] = (
            (latitude >= south) & (latitude <= north)
            & (longitude >= region["west"]) & (longitude <= east)
        )
    return masks
