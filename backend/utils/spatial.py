#%%
from shapely.geometry import Polygon


# =====================================================
# GRID CONFIG
# =====================================================

LATITUDE_STEP = 0.00135
LONGITUDE_STEP = 0.00165


# =====================================================
# CANONICAL CELL RESOLVER
# =====================================================

def latlon_to_cell(lat, lon):

    lat_idx = round(
        lat / LATITUDE_STEP
    )

    lon_idx = round(
        lon / LONGITUDE_STEP
    )

    return f"{lat_idx}_{lon_idx}"


# =====================================================
# CELL BOUNDARIES
# =====================================================

def cell_boundaries(cell_id):

    lat_idx, lon_idx = map(
        int,
        cell_id.split("_")
    )

    min_lat = lat_idx * LATITUDE_STEP
    max_lat = min_lat + LATITUDE_STEP

    min_lon = lon_idx * LONGITUDE_STEP
    max_lon = min_lon + LONGITUDE_STEP

    return {

        "min_lat": min_lat,
        "max_lat": max_lat,

        "min_lon": min_lon,
        "max_lon": max_lon
    }


# =====================================================
# CELL CENTER
# =====================================================

def cell_center(cell_id):

    bounds = cell_boundaries(
        cell_id
    )

    center_lat = (
        bounds["min_lat"]
        +
        bounds["max_lat"]
    ) / 2

    center_lon = (
        bounds["min_lon"]
        +
        bounds["max_lon"]
    ) / 2

    return (
        center_lat,
        center_lon
    )


# =====================================================
# CELL POLYGON
# =====================================================

def cell_polygon(cell_id):

    bounds = cell_boundaries(
        cell_id
    )

    polygon = Polygon([

        (
            bounds["min_lon"],
            bounds["min_lat"]
        ),

        (
            bounds["max_lon"],
            bounds["min_lat"]
        ),

        (
            bounds["max_lon"],
            bounds["max_lat"]
        ),

        (
            bounds["min_lon"],
            bounds["max_lat"]
        ),

        (
            bounds["min_lon"],
            bounds["min_lat"]
        )
    ])

    return polygon


# =====================================================
# ENSURE CELL EXISTS
# =====================================================

def ensure_cell_exists(

    cursor,

    cell_id
):

    cursor.execute(
        """
        SELECT 1
        FROM cells
        WHERE cell_id = %s
        """,
        (cell_id,)
    )

    exists = cursor.fetchone()

    if exists:
        return

    bounds = cell_boundaries(
        cell_id
    )

    center_lat, center_lon = (
        cell_center(cell_id)
    )

    polygon = cell_polygon(
        cell_id
    )

    cursor.execute(
        """
        INSERT INTO cells (

            cell_id,

            center_lat,
            center_lon,

            min_lat,
            max_lat,

            min_lon,
            max_lon,

            geom
        )

        VALUES (

            %s,

            %s, %s,

            %s, %s,

            %s, %s,

            ST_GeomFromText(%s, 4326)
        )
        """,
        (
            cell_id,

            center_lat,
            center_lon,

            bounds["min_lat"],
            bounds["max_lat"],

            bounds["min_lon"],
            bounds["max_lon"],

            polygon.wkt
        )
    )