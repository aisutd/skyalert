"""Where each pixel of a spread-model tile is on Earth.

The training tiles (Next Day Wildfire Spread) were exported from Earth Engine on its default
WGS84 degree grid at scale 1000 m: every pixel is 0.0089831528° tall and wide, the same in
degrees on both axes. The stored arrays are south-up (row 0 = south edge, column 0 = west edge).
See docs/spread_forecast_plan.md, "How the training tiles were made".

This is the only module that knows about orientation and pixel geometry. Layer builders ask a
TileGrid for pixel centres (`lats`, `lons`) and return arrays in the same layout, so the model
always sees its inputs the way it was trained.
"""

import math
from dataclasses import dataclass

import numpy as np

# Earth Engine's metres per degree at the equator for EPSG:4326, so scale 1000 m = this many degrees
PIXEL_DEG = 1000 / 111319.49079327357
TILE_SIZE = 64
# The fire the tile is built around sits in this row and column (training tiles were centred on a
# fire pixel at array index [31, 31])
CENTER_INDEX = 31
# Stored training arrays have the south edge in row 0. See the plan's "Open checks".
ROW0_IS_SOUTH = True

GEOMETRIES = ("degree", "km")


@dataclass(frozen=True)
class TileGrid:
    """A 64 x 64 tile on a regular lat/lon grid anchored at (0°, 0°)."""

    lat_step: float  # degrees per pixel, north-south
    lon_step: float  # degrees per pixel, east-west
    first_lat_index: int  # global index of the southernmost pixel row: its south edge is index * lat_step
    first_lon_index: int  # global index of the westernmost pixel column
    row0_is_south: bool = ROW0_IS_SOUTH
    size: int = TILE_SIZE

    @classmethod
    def around(cls, lat, lon, geometry="degree", row0_is_south=ROW0_IS_SOUTH):
        """The tile whose pixel [CENTER_INDEX, CENTER_INDEX] contains (lat, lon).

        geometry "degree" matches the training export: square in degrees, so narrower than
        64 km east-west away from the equator. "km" makes pixels about 1 km wide as well, by
        stretching the longitude step with 1 / cos(latitude); it exists to test which one fits.
        """
        if geometry not in GEOMETRIES:
            raise ValueError(f"geometry must be one of {GEOMETRIES}, got {geometry!r}")
        if not (-90 < lat < 90 and -180 <= lon <= 180):
            raise ValueError(f"not a valid location: {lat}, {lon}")
        lat_step = PIXEL_DEG
        lon_step = PIXEL_DEG if geometry == "degree" else PIXEL_DEG / math.cos(math.radians(lat))
        center_lat_index = math.floor(lat / lat_step)
        center_lon_index = math.floor(lon / lon_step)
        # Rows counted from the south edge: the centre pixel is CENTER_INDEX rows from row 0 when
        # row 0 is south, and CENTER_INDEX rows from row 0 (the north edge) otherwise
        rows_south_of_center = CENTER_INDEX if row0_is_south else TILE_SIZE - 1 - CENTER_INDEX
        return cls(
            lat_step=lat_step,
            lon_step=lon_step,
            first_lat_index=center_lat_index - rows_south_of_center,
            first_lon_index=center_lon_index - CENTER_INDEX,
            row0_is_south=row0_is_south,
        )

    # Global pixel indices along each axis, south to north and west to east
    @property
    def _lat_indices(self):
        return self.first_lat_index + np.arange(self.size)

    @property
    def _lon_indices(self):
        return self.first_lon_index + np.arange(self.size)

    def _rows_from_south(self, array):
        """Turn an array whose row 0 is south into model orientation (or back: it's its own inverse)."""
        return array if self.row0_is_south else array[::-1]

    @property
    def lats(self):
        """Latitude of every pixel centre, (64, 64), in model orientation."""
        column = (self._lat_indices + 0.5) * self.lat_step  # south to north
        return self._rows_from_south(np.repeat(column[:, None], self.size, axis=1))

    @property
    def lons(self):
        """Longitude of every pixel centre, (64, 64), in model orientation."""
        row = (self._lon_indices + 0.5) * self.lon_step  # west to east
        return np.repeat(row[None, :], self.size, axis=0)

    @property
    def bounds(self):
        """(south, west, north, east) of the tile's outer pixel edges, in degrees."""
        south = self.first_lat_index * self.lat_step
        west = self.first_lon_index * self.lon_step
        return (south, west, south + self.size * self.lat_step, west + self.size * self.lon_step)

    @property
    def center(self):
        """(lat, lon) of the centre of pixel [CENTER_INDEX, CENTER_INDEX]."""
        return float(self.lats[CENTER_INDEX, CENTER_INDEX]), float(self.lons[CENTER_INDEX, CENTER_INDEX])

    def cell_of(self, lats, lons):
        """Row and column (model orientation) of the pixel containing each point; -1 if outside."""
        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)
        from_south = np.floor(lats / self.lat_step).astype(np.int64) - self.first_lat_index
        cols = np.floor(lons / self.lon_step).astype(np.int64) - self.first_lon_index
        rows = from_south if self.row0_is_south else self.size - 1 - from_south
        inside = (from_south >= 0) & (from_south < self.size) & (cols >= 0) & (cols < self.size)
        return np.where(inside, rows, -1), np.where(inside, cols, -1)

    def rasterize(self, lats, lons):
        """A float32 (64, 64) mask: 1 in every pixel that contains at least one point, else 0."""
        mask = np.zeros((self.size, self.size), dtype=np.float32)
        rows, cols = self.cell_of(lats, lons)
        inside = rows >= 0
        mask[rows[inside], cols[inside]] = 1.0
        return mask

    def to_north_up(self, array):
        """A model-orientation array turned so row 0 is north, for drawing on a map."""
        array = np.asarray(array)
        return array[::-1] if self.row0_is_south else array
