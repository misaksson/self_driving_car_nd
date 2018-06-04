**Path planning**

The goals of this project are the following:
* Implement a path planner to drive a vehicle in high way traffic.
* Drive smoothly by not exceeding limits for acceleration and jerk.
* Drive near speed limit when possible. Never exceed speed limit.
* Overtake slower vehicles when possible.
* Behave nicely, e.g. don't collide with other vehicles. :)

Please see [Rubrics Points](https://review.udacity.com/#!/rubrics/1020/view) for further details about what's expected in this project.

## Project description
Implement a path planner that drives a vehicle on a high way with other traffic in a simulator environment. The simulator provides data about the localization of ego vehicle and detections of all other vehicles traveling in the same direction. The path planner provides as output a trajectory of coordinates representing the positions where the ego vehicle should be at each time-step. The path planner is allowed to update the trajectory a few times a second.

## Git branches
There are code on two branches for this project. On the [master](https://github.com/misaksson/self_driving_car_nd/tree/master/term3/P11-Path-Planning) branch the performance of the vehicle seems to be quite stable, and this is what's submitted for project review. Then there is a [work-in-progress](https://github.com/misaksson/self_driving_car_nd/tree/WIP-path-planning-prepare-lane-change/term3/P11-Path-Planning) branch (WIP), where the path planner is made more advanced such that it's able to adjust the speed before making a lane change. Although this approach looks promising, it would take more work than I'm prepared to spend on this project to make it good enough for project review. What remains is mainly additional tuning of cost functions and code refactoring. At this point the algorithm makes far too many mistakes, and the added code is a bit of a hack. Despite that, through out this report, the implementation on the WIP branch will be referenced to when applicable.

## Path planning
The path planner implemented in this project generates a lot of trajectories, then it applies cost functions to evaluate each of the trajectories and find the best one. But first, lets describe how a trajectory is calculated.

### Trajectory calculations
The trajectories are either starting from ego state or some position on the previous trajectory. In the later case, an ego state is derived from that point on the previous trajectory by calculating yaw-angle and speed from the Cartesian trajectory coordinates. The end of the trajectory is always set to be in the direction of the road, at some lateral coordinate d and longitudinal coordinate s ahead of start ego state, where s and d are Frenet coordinates. A spline library is then employed to create a smooth trajectory from the ego state [x, y, yaw-angle], to [s, d, yaw-angle<sub>road</sub>]. But to get the desired start and end yaw-angles in the spline, two additional points are needed. The first is calculated behind of ego state, using the yaw-angle, and the second is similarly choosen ahead of the target [s, d] by offsetting the s coordinate a few meters. The Frenet coordiantes is then transformed to Cartesian global coordinates, then transformed again, but this time to the local coordinate system of ego state. In the local coordinate system the coordinates are more or less along the x-axis (if not badly chosen target position), which ensures that a spline will fit nicely. This would not always be the case in global coordinates. Then finally, using the spline, any number of coordinates can now be calculate on the trajectory. To get a distance between the coordinates that roughly equals the desired speed, a linear approximation of the spline is used from the start to the end position.

All based on this method, there are several trajectory calculators implemented with slightly different purpose.

#### Max acceleration
This [method](https://github.com/misaksson/self_driving_car_nd/blob/a28fab8bd60a8ec44bc5f64c2fc8e1e4f6de2896/term3/P11-Path-Planning/src/path/trajectory.cpp#L100-L197) calculates the trajectory by applying the max permitted acceleration (10 m/s<sup>2</sup>) and jerk (10 m/s<sup>3</sup>). This is currently only used when keeping the same lane in order to reach the speed limit as quickly as possible. For more information about how the speed is calculated, please see [this comment](https://github.com/misaksson/self_driving_car_nd/blob/a28fab8bd60a8ec44bc5f64c2fc8e1e4f6de2896/term3/P11-Path-Planning/src/path/trajectory.cpp#L161-L176) in the code.

#### Adjust speed
This [method](https://github.com/misaksson/self_driving_car_nd/blob/a28fab8bd60a8ec44bc5f64c2fc8e1e4f6de2896/term3/P11-Path-Planning/src/path/trajectory.cpp#L292-L343) is the one mainly used when generating trajectories. It makes it possible to specify delta values for the longitudinal coordinate s, lateral coordinate d and the speed. A constant acceleration is used to do the speed change, and there are nothing assuring that the calculated trajectory is valid in terms of max acceleration and jerk so that must be verified later on in the cost functions.

#### Constant speed
This [method](https://github.com/misaksson/self_driving_car_nd/blob/a28fab8bd60a8ec44bc5f64c2fc8e1e4f6de2896/term3/P11-Path-Planning/src/path/trajectory.cpp#L199-L249) keeps a constant speed and lateral position d, for a selected number of coordinates. This is typically used to extend another trajectory to the desired number of coordinates.

### Trajectory generation
The trajectories are generated blindly at each update, without using a lot of logic to first figure out what a reasonable trajectory (not) should look like. Actually, the only logic used at this step in my implementation is to find what the next intention might be, by only considering the current lane of ego vehicle, e.g. doing a left lane change is always reasonable unless the vehicle already is in the left most lane. In total, there are [9 such intentions implement](https://github.com/misaksson/self_driving_car_nd/blob/a28fab8bd60a8ec44bc5f64c2fc8e1e4f6de2896/term3/P11-Path-Planning/src/path/logic.h#L16-L24). For each valid intention, several trajectories are generated by applying different longitudinal distance and speed changes in the execution of the intention. The number of trajectories generated must be balanced by how much processing power there is available to generate and evaluate all the trajectories.

### Prediction of other vehicles
The trajectories of other vehicles must be predicted for comparison with the ego vehicle trajectories.

#### Prediction using Frenet coordinates
The predictions are always made from the current state of the other vehicle, so that's the given start state provided to the trajectory calculator. The end state ís calculated by first transforming the given Cartesian velocities (vx, vy) into Frenet velocities (vs, vd). Then prediction of ongoing lane changes are done by looking at vd. When abs(vd) is above a threshold, then an lane change is assumed to be in the transition. The target d for the predicted trajectory is finally set to the center of predicted lane, and the target s of the trajectory depends on the remaining d transition and vd, but is at most 3 seconds * vs. The speed of other vehicles is assumed to be constant.

#### Prediction using Cartesian coordinates
The transformation to Frenet coordinates are not very reliable, and although I use a number of coordinates along the (vx, vy) vector to get a more robust (vs, vd) vector this sometimes fails, which is determined by a [sanity check](https://github.com/misaksson/self_driving_car_nd/blob/a28fab8bd60a8ec44bc5f64c2fc8e1e4f6de2896/term3/P11-Path-Planning/src/vehicle_data.h#L74-L83). As a fallback method, there is prediction based on the Cartesian velocities. This prediction is however really bad in curves, and frequently force ego vehicle into doing unexpected lane changes.

### Cost functions
Each generated trajectory is then processed by cost functions that adds up a scalar cost value that represents how good a trajectory is, i.e. the lower the cost, the better trajectory. Here's some examples of cost functions that I've implemented:
- Below the speed limit a cost is added that is proportional to the deviation.
- Above the speed limit a fixed, very high cost is added, effectively removing all such trajectories.
- Driving in the same lane as a slow going vehicle. A proportional cost is added similarly to when ego vehicle drives slowly.
- Driving close to other vehicles is punished by a cost factor that is inversely proportional to the shortest distance between the ego trajectory and any of the other vehicles predicted trajectories.
- Collisions, that is if the shortest distance is below a threshold of 3 m, then a fixed very high cost is added, effectively removing all such trajectories.
- When changing intention, such as when aborting a lane change, then a fixed cost is added.

In total, there are [14 cost functions](https://github.com/misaksson/self_driving_car_nd/blob/a28fab8bd60a8ec44bc5f64c2fc8e1e4f6de2896/term3/P11-Path-Planning/src/path/cost.h#L56-L184) implemented for this project. Each cost function is kept very simple, and typically only checks one aspect of the trajectory.

#### Tuning of cost functions
There are implicitly three groups of costs in my implementation:
1. Cost functions dealing with events that must not happen, e.g. collisions.
2. Cost functions controlling how the vehicle moves forward.
3. Cost functions that makes the vehicle move smoothly.

The highest costs are logically assigned to cost functions of the first group. These costs are sorted by importance and then assigned fixed costs that differ by some factor of 10. This assures that no trajectory where any such event happens is chosen, as long as there are other alternatives.

The middle range of cost values are distributed to the second group controlling how the vehicle moves forward. In this group, there are cost functions for keeping target speed, avoiding driving in a slow lane etc. which are weighted by other cost functions that prevents the vehicle from doing unnecessarily lane switch or taking dangerous paths. The costs in this group are mainly proportional to the error, and was tuned by running the simulator and adjust the cost factors until reaching the desired behavior. To help in this, each individual cost of all trajectories was output to a log file along with an estimated simulator time-stamp, which made it possible to investigate why one trajectory was chosen over another whenever there was a strange behavior showing up in the simulator. A visual tool processing the logs and presenting all the trajectories for example color coded by there cost value would have been great in this tuning process but there was no time for developing that.

The third group aims to select the smoothest trajectory when there are several performing about the same regarding the other more important aspects. In this group belongs cost functions for acceleration, jerk and yaw-rate, which produce cost values some factor 10 below the cost functions of group 1 and 2.

## Final result

### Master branch
On the master branch, the algorithm is quite stable and rarely produces mistakes. It's somewhat passive though and will always get stuck behind slower vehicles when the neighboring lane is occupied by other vehicles. In other words, there is no prepare for lane change implemented.

Here's a Youtube video showing a simulation with the master branch:
<a href="http://www.youtube.com/watch?feature=player_embedded&v=_XiLpjaC1Ik" target="_blank"><img src="http://img.youtube.com/vi/_XiLpjaC1Ik/0.jpg" alt="project_video output" width="640" height="360" border="10" /></a>

### Prepare for lane change
On the WIP-path-planning-prepare-lane-change branch, the algorithm do occasional mistakes. I was however able to record a simulation where it behaves somewhat okay. Note that the lane changes often are planned further ahead on the trajectory, after first adapting the speed and/or longitudinal position to the traffic in that lane.

Here's a Youtube video showing a simulation with the WIP branch:
<a href="http://www.youtube.com/watch?feature=player_embedded&v=osl_eeWCr44" target="_blank"><img src="http://img.youtube.com/vi/osl_eeWCr44/0.jpg" alt="project_video output" width="640" height="360" border="10" /></a>

### Maniac driving
Just for the fun of it, the target speed was increased from the speed limit of 50 MPH to 100 MPH. No other changes to the algorithm on the WIP branch was made.

Here's a Youtube video showing some maniac driving:
<a href="http://www.youtube.com/watch?feature=player_embedded&v=CZFgzrSPgDM" target="_blank"><img src="http://img.youtube.com/vi/CZFgzrSPgDM/0.jpg" alt="project_video output" width="640" height="360" border="10" /></a>

## Reflections
### Cost functions
I really like the approach of using cost functions to deal with advanced logical problems. It's so much easier and safer to define simple logic that looks at some aspect of a given/proposed solution, than trying to define a master-mind solution up front. I think it's safer since bad solutions are expected and will be removed by the cost functions, hence if there are possible corner cases in the algorithm, then the output solutions affected by them will effectively be discarded.

One might however have some concerns regarding the amount of computations needed in this algorithm, and I don't think there is anyone denying that an up front logical solver would be more efficient. This algorithm is however not needed to run very frequently, at least not every frame, and it's very easy to parallelize. There are also many ways to implement it more effectively than what I've done in this project. Here's some possible improvements that reduce the number of computations:
* Generate a course set of trajectories that are processed by the cost functions, then in the next step, generate additional trajectories near by the most promising trajectories.
* Abort as soon as the cost exceeds the lowest calculated so far.
* Improve this further by starting with the previous best trajectory to quickly get a low cost value.
* Improve this even further by sorting the cost functions by their average cost (or some other measure) divided by the computation time.

This type of algorithm is very intuitive to tune, so even if the intention is to use a different algorithm in the end, this may very well serve as an early reference model. The cost functions implemented may then later on be reused as sanity checks in the final product.

### Bad predictions of other vehicles
When there are strange behavior showing up in the simulation, e.g. when the trajectory swap between lanes for each update, then that's typically caused by bad predictions of other vehicles. In particular this happens when the fallback method of using vx, vy is applied in curves. Some sort of object tracking would therefore be the single most valuable improvement if one was about to continue with this project.
