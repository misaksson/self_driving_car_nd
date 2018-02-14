**Kidnapped Vehicle Project**

_Note: A writeup is not necessary to pass this project according to the [Rubrics Points](https://review.udacity.com/#!/rubrics/747/view). This short project report is mainly aiming to help me remember what was achieved in this project._

The goals of this project are the following:

* Implement a particle filter to localize the ego vehicles position by using a map of landmarks and observations.
* Use the bicycle motion model in the prediction step.
* Reach the performance goals as defined in the [Rubrics Points](https://review.udacity.com/#!/rubrics/747/view).

[//]: # (Image References)
[image1]: ./writeup_images/result.png

## Project description
Localize the position of ego vehicle using a map of landmarks and LIDAR observations of landmarks. To initialize the filter, a single uncertain GPS measurement is also provided.

### Particle filter
#### General
A particle filter generates lots of particles, which are more or less qualified guesses of the system state at time k. Then it evaluates how good each particle is based on how well it fits actual measurements of the system state. The better a particle fits the measurements, the higher is the probability that it will be represented in next iteration (when estimating the system state at time k+1). The output from the filter can either be the best particle or some weighted average of all particles. Before next iteration, the state of each particle is predicted individually, and in this step there is typically also some randomness involved based on the uncertainty of the model, which allows for some particles to catch up with how the actual system state is changing while other particles gets lost. Obviously, the particles that end up with a more correct state will better fit the measurements of the next iteration.

#### This implementation of a particle filter

##### Initialization
A single, uncertain GPS measurement is used to generate the initial particles. The particles are randomly generated according to the normal distribution defined for the GPS sensor.

##### Particle weights
In this project, the measurements are LIDAR observations of landmarks, which are compared to known positions of landmarks provided in a map. The comparison results in a weight for each particle, which becomes higher the better its observations fit the actual landmarks in the map. The first step in the weight calculation is to transform the observations to map coordinates, then each observation is associated to the nearest-neighboring landmark in the map. The uncertainty of the observations is defined by a normal distribution with known standard deviations, making it possible to calculate the probability of an observation using the multivariate probability density function. The weight of an particle is finally calculated as the product of all individual observation probabilities.

##### Particle resampling
The likelihood that a particle will be represented next iteration depends on its weight. That is, the particle is randomly selected for next iteration with a probability equal to its normalized weight. The random selection is done with replacement, meaning that one particle may be spawn multiple times next iteration, which typically becomes the case when one particle have significant higher weight than the others.

##### Prediction step
The position and attitude of each particle is then predicted at the point in time for the next measurement. This is done using the bicycle motion model, the particles position and yaw-angle, and provided measurements of velocity and yaw-rate. The measurements are noiseless according to a comment in the main.cpp (provided by Udacity), which either could mean that there is no noise to counter for in the motion model, or that each measurement is noiseless samples without counting for the vehicle acceleration and yaw-acceleration between the measurements. I chose to go with the first interpretation, but did however add some process noise to each particle before doing the _noiseless_ motion model update to counter for the uncertainty in the previous state of the particle. Also, it would be pointless to have multiple particles spawn from a single parent during the resample step, without adding some noise in the prediction step.

##### Output
Either the best particle, i.e. the one with highest weight, or some weighted average can be used as estimate for the ego vehicle state. In this project the best particle is used.

## Result
The algorithm is ran in a simulator provided by Udacity, which displays the ego vehicle (blue car) driving around in a world as defined by the map of landmarks. The landmarks are drawn as black x-circles and the actual observations is the green laser beams. The best particle position is drawn as a blue circle and it's observations are the blue laser beams. The root-mean squared error of the best particle compared to ground truth values are also displayed in the simulator.

![alt text][image1]

A Youtube video showing the simulation:
<a href="http://www.youtube.com/watch?feature=player_embedded&v=l-a4iq9fqRs" target="_blank"><img src="http://img.youtube.com/vi/l-a4iq9fqRs/0.jpg" alt="project_video output" width="640" height="360" border="10" /></a>
