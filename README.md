# MEMS Navigation Dataset

This repository contains a dataset for evaluating navigation algorithms using MEMS-grade sensors. Many other such datasets either are collected in a fairly small scale laboratory or field setting (typical of the robotics and unmanned systems community), or they focus on high-end sensors (typical of the marine and aerospace communities). In contrast, this dataset splits the difference: it is collected on what might be the most ubiquitous type of sensor configuration --- the MEMS-grade IMU and GPS antenna on most modern cell phones --- and contains longer term trajectories comparable to the marine and aerospace communities. It's primary contribution it to provide a dataset that is both accessible and representative of real-world navigation problems for education as well as simulation of conditions of GPS/GNSS degradation, spoofing, and intermittent availability to enable research into alternative navigation techniques.

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
