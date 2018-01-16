**Extended Kalman Filter Project**

_Note: A writeup is not necessary to pass this project according to the [Rubrics Points](https://review.udacity.com/#!/rubrics/748/view). This short project writeup report is mainly aiming to help myself remember what was achieved in this project._

The goals of this project are the following:

* Track an objects position relative the ego vehicle.
* Use Kalman filter to fusion measurements from both RADAR and LIDAR sensors.
* Implement a standard Kalman filter to process LIDAR measurements.
* Implement an extended Kalman filter to process the non-linear RADAR measurements.
* Calculate the mean-square error to compare the trackers result with ground truth values.

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/748/view) individually and describe how I addressed each point in my implementation.

---
### Compiling
#### 1. Your code should compile.
Yes it does, at least on my machine running Linux. I did however, against recommendations, do some changes to the CMakeLists.txt in order to implement unit tests, and this was also only verified to work on my machine.

### Accuracy
#### 1. px, py, vx, vy output coordinates must have an RMSE <= [.11, .11, 0.52, 0.52] when using the file: "obj_pose-laser-radar-synthetic-input.txt which is the same data file the simulator uses for Dataset 1"
My implementation gives an RMSE of [0.0973 0.0855 0.4513 0.4399] on Dataset 1, which is better than required.

### Follows the Correct Algorithm
#### 1. Your Sensor Fusion algorithm follows the general processing flow as taught in the preceding lessons.
My implementation is very much according to what was described in the lessons. I even use lesson examples as reference in my unit tests to verify the implementation.

#### 2. Your Kalman Filter algorithm handles the first measurements appropriately.
The first measurement is used to initialize the x_state vector, regardless if it's a LIDAR measurment or RADAR, where the later needs to be converted from polar to cartesian coordinates. This is implemented in ./src/FusionEKF.cpp lines 53-75.

#### 3. Your Kalman Filter algorithm first predicts then updates.
The prediction step is done before the update step, as can be seen in ./src/FusionEKF.cpp lines 77-134.

#### 4. Your Kalman Filter can handle radar and lidar measurements.
The algorithm does actually depend on fusion of both RADAR and LIDAR measurements to pass the accuracy requirements above. It's however easy to test how good the algorithm perform on either one of the sensors individually by simply commenting out the other sensors update step (the prediction step can be kept). The Youtube links below shows the result when running the algorithm with each sensor individually as well as fused together.

### Code Efficiency
#### 1. Your algorithm should avoid unnecessary calculations.
The guidelines suggested under this rubric point was followed, but I did nothing else to optimize the code for processing performance.

## Project description
A single object is tracked using sensor fusion of object measurements given both from RADAR and LIDAR sensors. The tracker algorithm is implemented using a Kalman filter, which implements the update step differently for the both sensor types.

### LIDAR
The LIDAR measurements are provided in cartesian coordinates, making it straight forward to just use the standard Kalman filter design.

LIDAR does not provide speed measure, so the H matrix will look like this: [1 0 0 0; 0 1 0 0]

The variance is 0.0225 in both x and y, and those both measurements are uncorrelated, giving the measurement covariance matrix R = [0.0225 0; 0 0.0225].

#### Standard Kalman filter
The x state vector keeps track of current best estimate of the objects position and speed, e.g. px, py, vx, vy. The uncertainty of this estimate is in the state covariance matrix P. This is initialized to have some uncertainty for position, and very high uncertainty for speed, which is reasonable since the initial state will not provide a speed measure at all from either sensor. Perhaps the position uncertainty initially should depend on the sensor type, e.g. lower uncertainty for LIDAR and higher for RADAR, but it looks like the tracker zooms in quite fast on the actual path regardless.

For each measurement, the Kalman filter updates the x and P state. The new state is first predicted, and then updated using the new measurement (z).

##### Prediction step
For this application, the x state is predicted based on position and speed in previous estimate together with the time elapsed since last measurement (dt), which gives the state transition matrix:
F = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1]
Note that there is no acceleration in this prediction, instead it's provided in the process noise covariance matrix Q, where the acceleration noise is set to 9 (./src/FusionEKF.cpp line 103).

The predicted state then becomes:
x = F * x
P = F * P * F_transpose + Q

##### Update step
The update step blends the predicted state x, with the measurement z. This is done based on the uncertainty on the predicted state x given in the P matrix, and the uncertainty of the measurement given in the R matrix.

First the measurement error (y) between the measurement z and the state x is calculated:
y = z - H * x

Then the innovation covariance matrix S is calculated, which is used to calculate the Kalman gain K, that is, how much this measurement should influence the new state of x and P.

S = H * P * H_transpose + R
K = P * H_transpose * S_inverse
x = x + K * y
P = (I - K * H) * P

### RADAR
The RADAR measurements on the other hand are a bit more tricky to handle since it provides polar coordinates [rho, phi, rho_dot]. Since the angular velocity phi_dot not is provided by the RADAR, conversion to cartesian coordinates isn't possible without losing the speed information. The same goes for the RADAR measurement variance that is provided by the sensor manufacturer and put in the measurement covariance matrix R.
The RADAR sensor in this project have a variance of 0.09 for the magnitude rho, 0.0009 for the angle phi, and 0.09 for the speed rho_dot. The measurements are uncorrelated, giving a measurement covariance matrix:
R = [0.09 0 0; 0 0.0009 0; 0 0 0.09]

#### Extended Kalman filter
This is where the Extended Kalman filter comes in handy. Instead of converting the measurement to cartesian coordinates, the position and speed estimate vector (x state) is converted to polar coordinates using the non-linear function h(x)=[sqrt(px^2 + py^2), atan2(py, px), (px * vx + py * vy) / sqrt(px^2 + py^2)]. At a given x state, it's possible to calculate a linear approximation of the measurement function h(x) as a first order multivariate Taylor series expansion, which is called an Jacobian matrix (Hj). By calculating the error (y) in polar coordinates
y = z - h(x)
and using Hj as measurement matrix, the remaining calculations are then done using the same procedure as for the standard Kalman filter. Note however that the Hj matrix must be calculated for each RADAR measurement.

The prediction step is linear for both LIDAR and RADAR. In other applications however, the prediction step, that is the transition function f(x), may be non-linear, which is something that also is handled by the extended Kalman filter. This is done similarly to how the non-linear measurement function is handled in this application, by approximating the non-linear transition function f(x) by an Jacobian matrix Fj at current x state.


## Result
The algorithm is ran in a simulator provided by Udacity, which displays the tracked object as it moves around together with sensor measurements and position estimates. The simulator displays LIDAR measurements as red circles, RADAR measurements as blue circles and tracked position as green triangles. The root-mean squared error (RMSE) of the estimate compared to ground truth values are also calculated and displayed in the simulator.

In the table below are my results from Dataset 1, showing the RMSE when using both sensors in fusion, and also when only using measurements from each sensor individually.

|    | Fusion | RADAR  | LIDAR  |
|----|--------|--------|--------|
| px | 0.0973 | 0.2302 | 0.1473 |
| py | 0.0855 | 0.3464 | 0.1153 |
| vx | 0.4513 | 0.5835 | 0.6383 |
| vy | 0.4399 | 0.8040 | 0.5346 |

A Youtube video showing the simulation results are also available:
<a href="http://www.youtube.com/watch?feature=player_embedded&v=jMq1cpJ8J1M" target="_blank"><img src="http://img.youtube.com/vi/jMq1cpJ8J1M/0.jpg" alt="project_video output" width="640" height="360" border="10" /></a>
