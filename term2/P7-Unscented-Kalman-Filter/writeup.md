**Unscented Kalman Filter Project**

_Note: A writeup is not necessary to pass this project according to the [Rubrics Points](https://review.udacity.com/#!/rubrics/783/view). This short project report is mainly aiming to help me remember what was achieved in this project._

The goals of this project are the following:

* Track position, velocity, yaw angle and yaw rate of a bicycle.
* Sensor fusion of measurements from both RADAR and LIDAR sensors.
* Use the motion model Constant Turn Rate and Velocity magnitude (CTRV).
* Implement an Unscented Kalman filter to do the non-linear prediction and update.
* Calculate the mean-square error to compare the trackers result with ground truth values.
* Calculate the Normalized Innovation Squared (NIS) value for each measurement.
* Tune the process noise parameters std_a and std_yawdd.

[//]: # (Image References)
[image1]: ./writeup_images/nis_dataset1.png
[image2]: ./writeup_images/nis_dataset2.png
[image3]: ./writeup_images/path_dataset1.png
[image4]: ./writeup_images/path_dataset2.png

## Project description
A single object (bicycle) is tracked by means of the CTRV motion model using sensor fusion of object measurements given both from RADAR and LIDAR sensors. The Unscented Kalman Filter (UKF) is implemented to deal with the non-linear process and measurement models.

### Unscented Kalman filter
#### Prediction step
The UKF use sigma points to approximate the non-linear transition in the model from time k to k+1. The sigma points are carefully selected to represent the probability distribution of the model at time k. Each sigma point is transitioned to time k+1 by passing it through the non-linear function derived from the CTRV motion model. The mean and covariance of the sigma points at time k+1 is then calculate, which is the predicted state x<sub>k+1|k</sub> and covariance matrix P<sub>k+1|k</sub>.

The process noise, which is part of the non-linear function derived from the CTRV motion model, is handled by augmenting the state vector x and covariance matrix P before calculating sigma points. The process noise for the CTRV motion model is described by std_a and std_yawdd, which are parameters that are tuned in this project. The state vector x is just extended by two zeros, and the process noise matrix Q expands the bottom right part of the covariance matrix P.

#### Update step
For the measurement update step, the procedure is similar as the prediction step but now we use sigma points to approximate the non-linear measurement function. Instead of generating new sigma points for the predicted state at time k+1, we can take a short-cut and reuse the sigma points transitioned from time k to k+1. The sigma points are then transformed to the measurement space for calculation of the mean vector and covariance matrix. The mean vector is the predicted measurement z<sub>k+1|k</sub>, and by adding the measurement noise R to the covariance matrix we get the innovation covariance matrix S<sub>k+1|k</sub>.

To calculate the Kalman gain K<sub>k+1|k</sub>, we first need to calculate T<sub>k+1|k</sub> which is the cross-correlation between sigma points in state space and measurement space. The Kalman gain is then calculated as
K<sub>k+1|k</sub> = T<sub>k+1|k</sub> * S<sup>-1</sub>.

With the Kalman gain, the new state and covariance matrix can be calculated as
x<sub>k+1|k+1</sub> = x<sub>k+1|k</sub> + K * y, where y is the measurement error z - z_pred.
P<sub>k+1|k+1</sub> = P<sub>k+1|k</sub> - K * S<sub>k+1|k</sub> * K<sup>T</sup><sub>k+1|k</sub>

This procedure works both for linear (LIDAR) and non-linear (RADAR) measurements, but for linear measurements, the standard Kalman filter is a better choose simple because it requires less computations, although the result is the same.

### Parameter tuning
The measurement noise R<sub>RADAR</sub>, and R<sub>LIDAR</sub>, are provided by the sensor manufacturer and does not need to be tuned in this project. The process noise matrix Q, containing the standard deviation of acceleration (std_a) and yaw acceleration (std_yawdd), is however unknown. I choose to tune these parameters by optimizing the performance on the both data sets on which it will be used.

A tuning tool was implemented (./src/main_tuning.cpp) that skips the simulator and instead directly read measurements from a text file. The tuning iterates over different values of std_a and std_yawdd and calculates the root-mean squared error (RMSE) for each of the data sets. The parameters that overall gave lowest RMSE was std_a = 0.46 and std_yawdd = 0.54.

#### Normalized Innovation Squared
To further figure out if the noise parameters in the model are reasonable, the Normalized Innovation Squared (NIS) is calculated for both radar and lidar measurements. The expected outcome is that the NIS values should follow the chi-squared distribution, for which there are tables available. The figures below shows the NIS value for each measurement in relation to the chi-squared value representing 95% of the distribution.

![alt text][image1]
![alt text][image2]

The distribution looks quite reasonable and the actual amount of measurements below the 95% line is also roughly as expected:

|               | Radar | Lidar |
|---------------|-------|-------|
| __Dataset 1__ | 0.956 | 0.976 |
| __Dataset 2__ | 0.952 | 0.928 |

## Result
The algorithm is ran in a simulator provided by Udacity, which displays the tracked object as it moves around together with sensor measurements and position estimates. The simulator displays LIDAR measurements as red circles, RADAR measurements as blue circles and tracked position as green triangles. The root-mean squared error (RMSE) of the estimate compared to ground truth values are also calculated and displayed in the simulator.

In the table below are my results in terms of RMSE when using both sensors in fusion and also each sensor individually. The last column shows the required result when using sensor fusion on dataset 1 (see [Rubrics Points](https://review.udacity.com/#!/rubrics/783/view)).

<table style="width:100%">
  <tr>
    <th> Dataset 1 </th><th> Dataset 2 </th><th> Required </th>
  </tr>
  <tr>
    <td>
      <table>
        <tr><th>  </th><th>Fusion</th><th>RADAR</th><th>LIDAR</th></tr>
        <tr><td><b>px</b></td><td>0.0601</td><td>0.2080</td><td>0.1584</td></tr>
        <tr><td><b>py</b></td><td>0.0843</td><td>0.2249</td><td>0.1464</td></tr>
        <tr><td><b>vx</b></td><td>0.3561</td><td>0.4072</td><td>0.3925</td></tr>
        <tr><td><b>vy</b></td><td>0.2482</td><td>0.1967</td><td>0.2568</td></tr>
      </table>
    </td>
    <td>
      <table>
        <tr><th>Fusion</th><th>RADAR</th><th>LIDAR</th></tr>
        <tr><td>0.1002</td><td>0.1899</td><td>0.1533</td></tr>
        <tr><td>0.0584</td><td>0.2112</td><td>0.1372</td></tr>
        <tr><td>0.6654</td><td>0.5966</td><td>0.4417</td></tr>
        <tr><td>0.2016</td><td>0.2824</td><td>0.2927</td></tr>
      </table>
    </td>
    <td>
      <table>
        <tr><th>Fusion</th></tr>
        <tr><td>0.09</td></tr>
        <tr><td>0.10</td></tr>
        <tr><td>0.40</td></tr>
        <tr><td>0.30</td></tr>
      </table>
    </td>
  </tr>
</table>

The reason that the result is worse in dataset 2 is due to the lack of knowledge about the initial yaw angle, and now it's initialized in the opposite direction. I've tried increasing the initial yaw angle uncertainty in the P matrix but it didn't seem to fix this problem.

A Youtube video showing the simulation results:
<a href="http://www.youtube.com/watch?feature=player_embedded&v=VUxbho_pjRs" target="_blank"><img src="http://img.youtube.com/vi/VUxbho_pjRs/0.jpg" alt="project_video output" width="640" height="360" border="10" /></a>

The results was also plotted to make it a bit easier to compare estimated and actual position for each data set:
![alt text][image3]
![alt text][image4]

## Catch the run away car (bonus challenge)
There was also a bonus challenge in this project. The UKF filter is now used to track a run away car, and the application shall steer a hunting vehicle to intercept it. The extra challenge in this is to predict the position of the run-away vehicle ahead of time, such that the hunting vehicle, although it drives at the same speed, will be able to intercept it.

My solution to this problem (./src/main_cath_the_run_away_car.cpp) predicts the position of the run away car ahead of time based on current Kalman filter state vector and the assumption that the car has constant velocity and yaw rate. The formula to calculate this prediction was already derived in the UKF filter implementation. I guess one could set up an equation and solve at what time (or position) the hunter will be able to intercept, but I chose to go with an iterative solution that simply increase the prediction time in small steps until finding that position.

The initial uncertainty in the Kalman filter makes the hunting vehicle appear a bit shaky until it finds a good bearing. I didn't implement any solution to this but it might be a good idea to consider the uncertainty of the state vector, i.e. the values of covariance matrix P, before doing any bold move in some direction.

Here's a video showing my result for this problem:
<a href="http://www.youtube.com/watch?feature=player_embedded&v=ED2wU7Oew4M" target="_blank"><img src="http://img.youtube.com/vi/ED2wU7Oew4M/0.jpg" alt="project_video output" width="640" height="360" border="10" /></a>
