**Model Predictive Control Project**

The goals of this project are the following:
* Implement a Model Predictive Controller that makes a vehicle follow waypoints.
* Handle an extra latency of 100 ms in the control loop.
* Describe the implemented model in detail.
* Describe how the parameters of the model where chosen.

Please see [Rubrics Points](https://review.udacity.com/#!/rubrics/896/view) for further details about what's expected in this project.


## Project description
Implement a controller that use Model Predictive Control (MPC) to drive a vehicle around a track in a simulator environment. The simulator provides the vehicles current position, orientation and velocity, as well as a few waypoints ahead of the vehicle that describes the track. An extra delay of 100 ms is also implemented in the control loop to mimic a real system where input sensor readings and output actuations typically not are disposed/applied without some delay.

## Model Predictive Control
The controller implemented in this project is based on a motion model that is used to predict how the vehicle would proceed forward when given a sequence of steering and throttle actuations to be carried out at specific time-steps. The predicted path is then compared to a target path, where the crosstrack error (cte) and orientation error (epsi) is calculated and accumulated for all time-steps in the prediction. The accumulated cte and epsi are finally evaluated by a cost function, that also looks at other aspects of the predicted path such as the actuations being applied, in order to provide a scalar cost value that defines how well this prediction suits the desired behavior of the controller. Given the motion model and the cost function, an optimizer is employed to find the predicted path that have the lowest cost. The very first actuations in the most cost effective prediction is then the only output from the controller at current time-step. For the next time step, the optimization process is repeated, since there is a new input state vector.

### Motion model
A motion model predicts how the state vector of a vehicle changes from time-step k to k+1. The motion model implemented in this project have a state vector that contains the vehicle position (x, y), velocity (v) and yaw-angle (psi). Input actuations of a vehicle typically consist of throttle, break and steering-angle. For simplicity, the throttle and break is a single actuator in this project, which directly maps to acceleration (a). The yaw-rate (psi_dot) is calculated from the steering-angle (delta), current velocity (v), and a constant Lf representing the distance from the front axle to the center of gravity.

Here's what the implemented motion model looks like:

```
dt = time[k+1] - time[k]
x[k+1] = x[k] + v[k] * cos(psi[k]) * dt
y[k+1] = y[k] + v[k] * sin(psi[k]) * dt
psi_dot[k] = (v[k] / Lf) * delta[k]
psi[k+1] = psi[k] + psi_dot[k] * dt
v[k+1] = v[k] + a[k] * dt
```

Note that this motion model doesn't make use of the acceleration in the translation part, hence it's a bit less accurate than the motion model previously implemented in the [Kidnapped-Vehicle project](../P8-Kidnapped-Vehicle/).

The motion model also ignores dynamics such as tire forces (slip angle and slip ratio), suspensions, mass, gravity, wind-resistance etc. Including such dynamics in  the model is considered an advanced topic far beyond the scope of this project.

Using a simple motion model might actually be beneficial as long as it's sufficient for the given task. A simple motion model is
* Easier to migrate to other vehicle types.
* Easier to maintain with simpler code and less parameters.
* Faster to process.

### Target path
The simulator provides waypoints representing the track nearby the current position of the vehicle. To be able to compare a predicted path with the target path, the waypoints are first transformed to the local coordinate system of the vehicle, then a polynomial is fitted. This makes it possible to at each timestep k of the predicted path also calculate cte and epsi.

```
cte[k] = f(x[k]) - y[k],
    where f() is the fitted polynomial.
epsi[k] = atan(f'(x[k])) - psi[k]
    where f'() is the derivative of the fitted polynomial.
```
A short road segment is typically possible to represent well enough using a third degree polynomial. Note however that using the local coordinate system of the vehicle really simplifies the calculation compared to going with global coordinates. In local coordinates, the road segment to follow stretches out more or less in parallel to the x-axis, such that a polynomial fits nicely. In global coordinates on the other hand, it's easy to find road segments that wouldn't be possible to represent using a standard polynomial, such as when there are multiple y-coordinates for a single x-coordinate. Also, the calculations in the motion model would be much more cumbersome to handle.

### Optimizer
The MPC controller use an optimizer to find the sequence of actuations that, according to the prediction, best follows the desired behavior of the system. In this project, the vehicle shall drive safely around the track, and as a bonus challenge, it shall also do this as fast as possible. This does of course involve keeping cte and epsi small, but there is some room for flexibility without leaving the race track, e.g. to find a straighter trajectory through curves. Another given is that the vehicle should maintain as high speed as possible. A maybe less obvious requirement might be that it shouldn't use actuations to aggressively or suddenly, and that is not just out of convenience for the passengers. My controller is actually unable to drive around the track other than at very low speeds without penalizing steering actuations. I believe this is caused by bad predictions from the motion model when the vehicle is turning sharply. However, by pushing for a straighter path, the implemented motion model is more accurate and this doesn't become an issue. Another benefit of keeping a straighter path is that the controller, although unaware about the concept, still avoids loosing speed due to the tire forces applied when turning.

#### Cost functions
The requirements described above is defined by a cost function in the code, which is evaluated by the optimizer in it's search for the best prediction. The lower cost, the better prediction. Below is the cost function that is used in my final implementation.

```
cost = 0
for each predicted state i = 1..N:
    cost += 5000 * cte[i]^2
    cost += 1000 * epsi[i]^2
    cost += 150 * (v[i] - target_v)^2,
        where target_v is 47 meter per second (105 MPH).
for each predicted actuation i = 1..N-1:
    cost += 45000000 * delta[i]^2
```

Note that the absolute value of each factor doesn't matter, instead it's all about finding a good balance between the costs. The cost factor applied on delta may seem huge, and it is, but the delta signal is also some magnitude smaller the other values.

##### Tuning
The cost factors shown above has been tuned manually, typically adjusted by a factor of 10 until reaching a good value to further zoom in on. Below are a few observations that I did during the tuning. It also includes the parameters for the prediction time step (dt) and the number of prediction steps (N), which are very much involved when tuning the cost factors.

* **delta vs velocity**: The cost of doing delta actuation must be balanced to the cost of not maintaining the target velocity, such that the optimizer doesn't find it more cost effective to stop than to continue through a curve. It was actually a surprisingly narrow margin between having a controller that completely stops in the curve to one that is very aggressive and barley slows down at all.
* **cte vs epsi (and/or delta)**: As the target velocity is increased, the problems of having a simple motion model becomes more and more apparent. The controller seems to turn more than it is predicting, making the vehicle wobble due to cte overshoots, which in particular happens at straighter parts of the track. To counter this, the cost of cte must be balanced by the cost of epsi, where the first will strive to get a more aggressive path towards the road center and the later basically strives to maintain current distance to the road center. Increasing the cost of delta will also reduce the cte overshoots.
* **dt**: The delta time for the prediction steps was first set to 50 ms. This worked fine until the 100 ms latency was activated. The problem is that the controller framework doesn't implement pipelining of actuations, meaning that the extra latency will increase the total time between actuations to about 117 ms (measured on my PC). If the controller then use a prediction steps of 50 ms, then it will come up with a sequence of actuations that in reality isn't possible to execute, hence produce an unrealistic prediction. Instead, the delta time was set to allow the controller to as closely as possible mimic the actuations that are possible in reality. In the final implementation, the actuation period is continuously measured and a low-pass filtered average is calculated and used as delta time.
* **N**: The number of prediction steps must be chosen such that the prediction looks far enough to be able to prepare for curves in time, still not so far that the uncertain predictions at longer distance influence the cost too much. When using a dt of about 117 ms and a target speed of 105 MPH, I found that 13 prediction steps was a good value. A smaller number doesn't let the controller prepare for curves in time. On the other hand, a larger number gives, due to the simple motion model, too bad predictions when driving in curves. I did actually do an attempt to use a variable number of prediction steps, based on the vehicle speed and/or turn rate. N would then be increased on straighter parts of the track, where the vehicle speeds up, and decreased in curves. Although I think this approach might give a better result in the end, I didn't pursue it after realizing that the cost functions had to be tuned individually for each number of prediction steps.

#### Constraints
A part from the motion model and the cost function, the optimizer is also given a number of constraints. The permitted ranges for the predicted state variables and actuators are specified, as well as constraints on how the motion model optimally should evaluate.

In this project, the optimizer are free to choose whatever values it like for the predicted states, as long as it follows the motion model. There are however physical limitations on the possible actuation signals implemented in the simulator:

```
delta_min = -25 degrees
delta_max = 25 degrees
a_min = -1 m/s^2
a_max = 1 m/s^2
```

The constraints put on the motion model evaluation does essentially say that the prediction should start from the current state, then each state transition from time step k to k+1 should be spot on the motion model prediction.

### Latency
To mimic the behavior of a real system, an extra latency of 100 ms is added to the control loop. Since the controller framework doesn't implement pipelining of actuations, the effect of this latency is just that the previously issued actuation will be active for an additional 100 ms. My solution to this problem is to predict the state of the vehicle after 100 ms, by using the same motion model as described above. One should perhaps predict some additional milliseconds due to actual latency in the simulator, but I didn't do this. The predicted state is then used as initial state for the MPC solver.

Another solution to the latency problem would be to let the MPC solver do the job of predicting the state after 100 ms. This can be accomplished by adding the ongoing (already decided) actuations as extra constrains for the optimizer. This would however either require that the latency is an multiple of the dt parameter, or that the implementation allows the constrained prediction steps to use another dt. This is obviously more messy then to just predict the vehicle state before calling MPC solver.

## Implementation
### Target Path
The class [Path](./src/path.h) calculates a third degree polynomial representing the track in vehicle local coordinates. For convenience, it also provides the vehicle state in local coordinates, and also the track reference displayed in yellow by the simulator.

### Model Predictive Control
The class [MPC](./src/MPC.h) implements two methods that are essential for this project.
* *Predict()* use the motion model to predict the vehicle state after the 100 ms latency period.
* *Solve()* use the Ipopt library to solve the nonlinear optimization problem of finding the best prediction.

### Polynomial
The class [Polynomial](./src/polynomial.h) implements helper methods to fit, evaluate and calculate the derivative of an polynomial.

### Main
The [main.cpp](./src/main.cpp)
* Interacts with the simulator using a websocket interface.
* Use the classes described above to calculate the next actuations.
* Implements the 100 ms extra latency, just before dispatching the actuation message to the simulator.

## Final result
The controller was manually tuned to achieve high speed. In the final configuration, the target velocity is set to 47 m/s (~105 MPH), and it's actually not far from reaching this speed at some parts of the track. The cost functions will however make the vehicle slowdown to about 75 MPH when going through the steeper curves. This tuning is quite extreme tough, and the controller does fail sometimes which I think might be due to real-time problems of running both the simulator and the controller asynchronously on the same PC, that also do lots of other stuff in the background. I recommend running the simulator at lowest/fastest graphic quality to get less trouble with this.

Here's a Youtube video showing the simulation:
<a href="http://www.youtube.com/watch?feature=player_embedded&v=b0Iha1X5WBs" target="_blank"><img src="http://img.youtube.com/vi/b0Iha1X5WBs/0.jpg" alt="project_video output" width="640" height="360" border="10" /></a>
