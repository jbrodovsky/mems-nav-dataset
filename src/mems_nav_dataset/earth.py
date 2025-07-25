import numpy as np

# Earth's rotation rate in radians per second
EARTH_ROTATION_RATE: float = 7.2921159e-5  # rad/s
# Earth's magnetic north pole latitude, degrees (2025, International Geomagnetic Reference Field)
MAGNETIC_NORTH_LATITUDE: float = 80.8
# Earth's magnetic north pole longitude, degrees (2025, International Geomagnetic Reference Field)
MAGNETIC_NORTH_LONGITUDE: float = -72.8
# Earth's magnetic reference radius, meters (2025, International Geomagnetic Reference Field)
MAGNETIC_REFERENCE_RADIUS: float = 6371200.0
# Earth's magnetic field strength (B0), teslas (2025, International Geomagnetic Reference Field)
MAGNETIC_FIELD_STRENGTH: float = 3.12e-5
# Rough conversion factor from meters to degrees for latitude/longitude via nautical miles
METERS_TO_DEGREES: float = 1.0 / (60.0 * 1852.0)
# Rough conversion factor from degrees to meters for latitude/longitude via nautical miles
DEGREES_TO_METERS: float = 60.0 * 1852.0
# Earth's equatorial radius in meters
EQUATORIAL_RADIUS: float = 6378137.0
# Earth's polar radius in meters
POLAR_RADIUS: float = 6356752.31425  # meters
# Earth's mean radius in meters
MEAN_RADIUS: float = 6371000.0  # meters
# Earth's eccentricity ($e$)
ECCENTRICITY: float = 0.0818191908425  # unit-less
# Earth's eccentricity squared ($e^2$)
ECCENTRICITY_SQUARED: float = ECCENTRICITY * ECCENTRICITY
# Earth's gravitational acceleration at the equator ($g_e$) in $m/s^2$
GE: float = 9.7803253359  # m/s^2, equatorial radius
# Earth's gravitational acceleration at the poles ($g_p$) in $m/s^2$
GP: float = 9.8321849378  # $m/s^2$, polar radius
# Somigliana's constant ($K$) for the Earth
K: float = (POLAR_RADIUS * GP - EQUATORIAL_RADIUS * GE) / (EQUATORIAL_RADIUS * GE)


def gravity(
    latitude: float | np.typing.NDArray, altitude: float | np.typing.NDArray
) -> float | np.typing.NDArray:
    """
    Calculate the gravitational acceleration at a given latitude and altitude.
    Returns gravitational acceleration in m/s^2.
    """
    lat_rad = np.radians(latitude)
    sin_lat = np.sin(lat_rad)
    g = (GE * (1.0 + K * sin_lat * sin_lat)) / (
        1.0 - ECCENTRICITY_SQUARED * sin_lat * sin_lat
    )
    g -= 3.08e-6 * altitude  # Adjust for altitude (approximation)
    return g


def wgs84_to_magnetic(latitude, longitude):
    """
    Calculate magnetic colatitude and longitude from WGS84 coordinates.
    Returns (magnetic_colatitude, magnetic_longitude) in degrees.
    """
    lat_rad = np.radians(latitude)
    lon_rad = np.radians(longitude)
    mag_lat_rad = np.radians(MAGNETIC_NORTH_LATITUDE)
    mag_lon_rad = np.radians(MAGNETIC_NORTH_LONGITUDE)

    cos_theta = np.sin(lat_rad) * np.sin(mag_lat_rad) + np.cos(lat_rad) * np.cos(
        mag_lat_rad
    ) * np.cos(lon_rad - mag_lon_rad)
    y = np.sin(lon_rad - mag_lon_rad) * np.cos(lat_rad)
    x = np.cos(mag_lat_rad) * np.sin(lat_rad) - np.sin(mag_lat_rad) * np.cos(
        lat_rad
    ) * np.cos(lon_rad - mag_lon_rad)

    mag_latitude = np.degrees(np.acos(cos_theta))
    mag_longitude = np.degrees(np.atan2(y, x))
    return mag_latitude, mag_longitude


def calculate_radial_magnetic_field(colatitude, radius):
    """
    Calculate the radial component of Earth's magnetic field using the dipole model.
    colatitude: radians, radius: meters
    Returns: radial component in teslas
    """
    return (
        -2.0
        * MAGNETIC_FIELD_STRENGTH
        * (MAGNETIC_REFERENCE_RADIUS / radius) ** 3
        * np.cos(colatitude)
    )


def calculate_latitudinal_magnetic_field(colatitude, radius):
    """
    Calculate the latitudinal component of Earth's magnetic field using the dipole model.
    colatitude: radians, radius: meters
    Returns: latitudinal component in teslas
    """
    return (
        -MAGNETIC_FIELD_STRENGTH
        * (MAGNETIC_REFERENCE_RADIUS / radius) ** 3
        * np.sin(colatitude)
    )


def calculate_magnetic_field(latitude, longitude, altitude):
    """
    Calculate the magnetic field using the Earth's dipole model in the local-level frame.
    Returns: numpy array [North, East, Down] in teslas
    """
    mag_colatitude, _ = wgs84_to_magnetic(latitude, longitude)
    mag_colatitude_rad = np.radians(mag_colatitude)
    radius = MAGNETIC_REFERENCE_RADIUS + altitude

    radial_field = calculate_radial_magnetic_field(mag_colatitude_rad, radius)
    lat_field = calculate_latitudinal_magnetic_field(mag_colatitude_rad, radius)

    # In the NED frame: [North, East, Down]
    b_vector = np.zeros(latitude.shape + (3,), dtype=np.float64)
    b_vector[:, 0] = radial_field
    b_vector[:, 1] = lat_field
    return b_vector


def magnetic_inclination(latitude, longitude, altitude):
    """
    Calculate the magnetic inclination (dip angle) at a given location.
    Returns: inclination angle in degrees
    """
    b_vector = calculate_magnetic_field(latitude, longitude, altitude)
    b_h = np.sqrt(b_vector[0] ** 2 + b_vector[1] ** 2)
    # Inclination: atan(Down / Horizontal)
    # Here, b_vector[2] is Down (which is always 0 in this model)
    return np.degrees(np.atan2(b_vector[2], b_h))


def magnetic_declination(latitude, longitude, altitude):
    """
    Calculate the magnetic declination (variation) at a given location.
    Returns: declination angle in degrees
    """
    b_vector = calculate_magnetic_field(latitude, longitude, altitude)
    # Declination: atan2(East, North)
    return np.degrees(np.atan2(b_vector[1], b_vector[0]))
