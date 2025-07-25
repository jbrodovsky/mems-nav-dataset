# MEMS Navigation Dataset
This repository contains a dataset and toolbox for evaluating navigation algorithms using MEMS-grade sensors. The dataset includes raw measurements from GPS and IMU sensors as well as calculated INS data. The toolbox provides utilities for processing and analyzing these measurements. The primary focus is on enabling the research of navigation in GPS/GNSS denied, degraded, or contested environments using low-cost MEMS sensors such as what you would find on a typical smartphone.

## Dataset Overview

Added 1Hz, 5Hz, and 10Hz data.

### Raw Data

These are the raw data files containing the measurements from the sensors used in the dataset. The files are in CSV format and include:

- **Accelerometer**: 3-axis acceleration measurements with gravity compensation
- **Barometer**: Barometric pressure measurements and computed relative altitude changes
- **Gravity**: 3-axis gravity vector measurements
- **Gyroscope**: 3-axis angular velocity measurements
- **LocationGPS**: WGS84 latitude, longitude, and altitude measurements
- **Magnetometer**: 3-axis magnetic field measurements
- **Orientation**: 3-axis orientation estimates (quaternion and Euler angles)

among others.

### Ground Truth

Full state loosely coupled UKF INS estimates using GPS position and velocity measurements as well as on-board computed barometric altitude and magnetic heading. This ground truth is considered accurate and valid as the positioning error is less than the GPS confidence accuracy.

