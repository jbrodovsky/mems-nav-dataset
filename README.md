# MEMS Navigation Dataset

This repository contains a dataset for evaluating navigation algorithms using MEMS-grade sensors. Many other such datasets either are collected in a fairly small scale laboratory or field setting (typical of the robotics and unmanned systems community), or they focus on high-end sensors (typical of the marine and aerospace communities). In contrast, this dataset splits the difference: it is collected on what might be the most ubiquitous type of sensor configuration --- the MEMS-grade IMU and GPS antenna on most modern cell phones --- and contains longer term trajectories comparable to the marine and aerospace communities. It's primary contribution it to provide a dataset that is both accessible and representative of real-world navigation problems for education as well as simulation of conditions of GPS/GNSS degradation, spoofing, and intermittent availability to enable research into alternative navigation techniques.

## Raw Data

These are the raw data files containing the measurements from the sensors used in the dataset. The files are in CSV format and include:

- **Accelerometer**: 3-axis acceleration measurements with gravity compensation
- **Barometer**: Barometric pressure measurements and computed relative altitude changes
- **Gravity**: 3-axis gravity vector measurements
- **Gyroscope**: 3-axis angular velocity measurements
- **LocationGPS**: WGS84 latitude, longitude, and altitude measurements
- **Magnetometer**: 3-axis magnetic field measurements
- **Orientation**: 3-axis orientation estimates (quaternion and Euler angles)

among others related to the device's sensors and state. The raw data files are not stored in this repository due to their size and are available upon request. Downsampled and time-synchronized versions of these files are included in the `input` directory.

## Processed Data

The raw files are processed in several different ways via the [strapdown-sim](https://github.com/jbrodovsky/strapdown-rs) simulations to produce the results in the following configurations: a baseline ground truth, scheduling degredation variations, and spoofing/jamming scenarios. 

- **Baseline**: Standard strapdown inertial navigation with GPS updates and no degradation
- **Combination**: `combo`, combination of degraded and intermittent GPS availability
- **Combination with duty cycle and hijacking**: `combo_duty_hijack`, combination of degraded and intermittent GPS availability with duty cycling and spoofing
- **Degradation with intermittent availability**: `degraded_5s`, degraded GPS measurements with intermittent availability
- **Degraded**: `degraded_fullrate`, GPS measurements are degraded with correlated noise and inflated covariance
- **Duty-cycle**: `duty_10on_2of`, GPS available for 10 seconds, then off for 2 seconds
- **Hijacking**: `hijack`, GPS measurements are spoofed for a fixed interval of time with a constant offset
- **Fixed-interval**: `sched_10s`, GPS updates every 10 seconds
- **Slow bias**: `slowbias`, GPS measurements are biased with a slow drift over time
- **Slow bias with roation**: `slowbias_rot`, GPS measurements are biased with a slow drift and the device is rotated
