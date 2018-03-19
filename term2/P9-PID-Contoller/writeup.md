**PID Controller Project**

The goals of this project are the following:

* Implement a PID controller that drives a vehicle when provided with cross-track error and velocity.
* Reflect on the effect of each component of the PID controller.
* Tune the controller to safely drive as fast as possible around a track.
* Describe the tuning process.

See [Rubrics Points](https://review.udacity.com/#!/rubrics/824/view) for details about what's expected in this project.

## Project description
Implement a controller, that use Proportional, Integral and Derivative (PID) components to drive a vehicle around a track in a simulator environment. The simulator provides current cross-track error and velocity as input to the controller, which then shall provide steering angle and throttle/break value.

## PID controller
Here's a reflection of each of the components of the PID controller.

### Proportional
The proportional part is the current error multiplied by a coefficient Kp. A controller only using the P component would look like this:

```
actuation[t] = -error[t] * Kp
```

A problem with such a controller is that it always more or less will overshoot the target value. How much it overshoots depends on the value of Kp and how stable the system is by it self. In this project, it might be possible to control the vehicle up to some velocity using only the P component. But above that velocity, there will be ever growing overshoots that ultimately makes the vehicle drive of the track and crash.

### Integral
The integral part accumulates all errors over time and then multiply by a coefficient Ki. A controller only using the I component would look like this:

```
actuation[t] = -accumulate(errors[0..t]) * Ki
```

This is to adjust for biases in the system. When steering a vehicle, it might be that the steering angle is offset by some value, or that the road is turning more in some direction. In this project, the vehicle drives around the track counter-clockwise, which should give a positive bias on the accumulated error value. I also noticed that the Udacity simulator [implements an additional steering bias of 1 degree](https://github.com/udacity/self-driving-car-sim/blob/term2_collection/Assets/1_SelfDrivingCar/Scripts/project_4/CommandServer_pid.cs#L58-L59).

### Derivative
The derivative part is the change in error since last measurement, multiplied by a coefficient Kd. A controller only using the D component would look like this:

```
actuation[t] = -(error[t] - error[t-1]) * Kd
```

This will effectively strive to maintain the current error. When the error is increasing, then it it will increase the countering actuation. When the error is decreasing, then it will reduce the actuation that counters the error. When combined with the P component, the D component will reduce the overshoots and help stabilize the system.

## Implementation
### PID controller
The PID controller is implemented just as it was taught in the lessons, see [./src/PID.h](https://github.com/misaksson/self_driving_car_nd/blob/master/term2/P9-PID-Contoller/src/PID.h) and [./src/PID.cpp](https://github.com/misaksson/self_driving_car_nd/blob/master/term2/P9-PID-Contoller/src/PID.cpp). The steer signal output by the controller is divided by the vehicle speed in an attempt to make a single PID controller work for different velocities.

### Twiddle tuning algorithm
To tune the PID controller, the Twiddle algorithm was implemented. I choose to implemented Twiddle as a class inheriting the PID class and extending it with tuning functionality, which mainly is implemented in the method [SetNextParams()](https://github.com/misaksson/self_driving_car_nd/blob/master/term2/P9-PID-Contoller/src/twiddle.cpp#L61-L126). The other methods added by the Twiddle class provides functionality to allow for interchangeably tuning of multiple PID controllers. When the Twiddle algorithm not is active, then it works just as the PID class.

### Crosstrack error evaluator
Multiple PID controllers are used to control the vehicle, but only one at a time. Which one to apply depends on how good the controller are currently working, which is measured by accumulating the cross-track error (CTE) over a time period. This is implemented in a class [CrosstrackErrorEvaluator](https://github.com/misaksson/self_driving_car_nd/blob/master/term2/P9-PID-Contoller/src/cte_eval.h) which defines six performance levels that ranges from DEFECTIVE to IDEAL.

#### Curve detector (failed implementation)
At first, the plan was to calculate the curvature of the road and let this decide which PID controller to apply. Although this at first seemed possible, e.g. by estimating the vehicles path using a motion model, and then put it in relation to how the CTE changes. My attempts in doing this did however result in almost negligible input from the motion model in comparison to the CTE changes, hence the CrosstrackErrorEvaluator described above.

### Vehicle Controller
#### Control modes
The class [VehicleController](https://github.com/misaksson/self_driving_car_nd/blob/master/term2/P9-PID-Contoller/src/vehicle_controller.h) implements a control mode for each performance level defined by the Cross-Track Error Evaluator. A control mode have two instances of the PID controller, one for steering and the other for throttle, accompanied by a predefined target velocity.

#### Tuning management
The Vehicle Controller also manages over the Twiddle tuning algorithm, essentially cycling through all PID controller instances and let them alternate there coefficients one at a time. The tuning is evaluated by measuring how long time it takes to travel around the track, where the mean + standard deviation of 7 laps is used as performance score/cost.

### Simple Timer
The class [SimpleTimer](https://github.com/misaksson/self_driving_car_nd/blob/master/term2/P9-PID-Contoller/src/simple_timer.h) wraps the timer functionality provided by the chrono standard library to provide the delta time. This is used together with velocity to estimate the traveled distance, which is used to count laps.

### Main
The [main.cpp](https://github.com/misaksson/self_driving_car_nd/blob/master/term2/P9-PID-Contoller/src/main.cpp) interacts with the simulator using a websocket interface and calls the controller and tuning code described above, to provide next steering and throttle value.

## Tuning process
### Initial attempts
The PID coefficients were initially set just as in an unrelated lesson example, and although very wobbly, this did actually make the vehicle drive around the track in speeds up to ~30 MPH. Those coefficients was then used as a starting point for the Twiddle algorithm.

### Safe mode
One problem with tuning the coefficients in the simulator is that when the vehicle crashes, then the simulator must be manually restarted. To work around this, a "safe mode" was implemented, in which the speed was reduced and a special PID controller instance that was good at recovering was activated. The crosstrack error was evaluated to decide when to switch to safe mode.

Safe mode was actually later on expended to multiple modes and used in the final controller, as described in the Implementation section.

### Performance score
One of the hardest parts when tuning this controller was to understand if one set of coefficients was better then another, i.e. to come up with a good and stable performance score. Early on, I accumulated the crosstrack error of all samples from 1 lap, which however quite often gave false improvements ("lucky shots"), that was almost impossible to beat. The number of laps were then increased, making the score a bit more stable.

After implementing safe-mode as described above, the performance score was changed to just measure how fast the vehicle is able to drive around the track. Indirectly however, due to the implementation of control modes that have different target speeds, this still tunes the controller towards a lower CTE. The number of laps was gradually increased, and the final score is calculated as the mean + standard deviation of 7 laps, trying to get a controller that is fast around the track with small variations.

### Multiple PID controllers
The tuning of multiple control modes is managed by the VehicleController class, which is cycling through all PID controllers, letting each Twiddle instance alternate all coefficients one at time, before continuing with next control mode.

### Overfitting
One problem with my approach of switching between multiple control modes is that, although unintentionally, each control mode might become specialized at handling one or a few parts of the track. By looking at the [final coefficients](https://github.com/misaksson/self_driving_car_nd/blob/master/term2/P9-PID-Contoller/src/vehicle_controller.cpp#L12-L56), this might actually be what has happened here, e.g. it's hard to see a pattern in how the coefficients are chosen. One way to prevent this would be to include other tracks in the tuning process.

### Final result
The controller was tuned to achieve fast lap times, and although the standard deviation was added in the cost function in an attempt to also make the controller produce more repeatable results, the lap times still vary somewhere between 50-60 seconds on my PC. I think that some of the variations might be due to the real-time aspects of having the controller run in a from the simulator asynchronous process, on a PC that, apart from the simulator, also do lots of other stuff in the background. As a quick test to somewhat confirm this theory, an additional delay of 10 ms was added to the control loop. Although the controller still manages to safely drive around the track, this delay did have an surprisingly negative effect on the performance. In segments of the track where it typically would reach speeds up to 70 MPH it now barley makes 50 MPH.

A Youtube video showing one of the faster laps (~52 seconds):
<a href="http://www.youtube.com/watch?feature=player_embedded&v=xDvOpZjt1aw" target="_blank"><img src="http://img.youtube.com/vi/xDvOpZjt1aw/0.jpg" alt="project_video output" width="640" height="360" border="10" /></a>
