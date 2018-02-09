/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

#include "particle_filter.h"
#include "map.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Set the number of particles.
  num_particles = 1000;

  // Initialize all particles to first position (based on estimates of x, y,
  // theta and their uncertainties from GPS) and all weights to 1.
  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) {
    particles.push_back({.id=i, .x=dist_x(gen), .y=dist_y(gen), .theta=dist_theta(gen), .weight=1.0});
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /* The velocity and yaw_rate is noiseless according to a comment in main.cpp.
   * The std_pos provided by main.cpp actually relates to GPS measurements, but
   * here it's instead added to the particle positions after motion model
   * update.
   *
   * TODO: Is it really correct to use the same noise in this prediction as in
   *       the initialization, which is based on the uncertain GPS measurement?
   * TODO: Why not use GPS to improve prediction?
   */
  const double eps = 0.0001;
  default_random_engine gen;
  normal_distribution<double> dist_x(0.0, std_pos[0]);
  normal_distribution<double> dist_y(0.0, std_pos[1]);
  normal_distribution<double> dist_theta(0.0, std_pos[2]);

  for (auto particle = particles.begin(); particle != particles.end(); ++particle) {
    const double yaw_angle = particle->theta;

    // Avoid division by zero
    if (fabs(yaw_rate) < eps) {
      // Driving straight
      particle->x += velocity * cos(yaw_angle) * delta_t;
      particle->y += velocity * sin(yaw_angle) * delta_t;
    } else {
      // Turning
      particle->x += (velocity / yaw_rate) * (sin(yaw_angle + (yaw_rate * delta_t)) - sin(yaw_angle)) + dist_x(gen);
      particle->y += (velocity / yaw_rate) * (-cos(yaw_angle + (yaw_rate * delta_t)) + cos(yaw_angle)) + dist_y(gen);
    }
    particle->theta += yaw_rate * delta_t + dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(const Map& map,
                                     std::vector<TransformedObservation>& observations) {
  for (int obsIdx = 0; obsIdx < observations.size(); ++obsIdx) {
    observations[obsIdx].landmarkIdx = TransformedObservation::invalidLandmarkIdx;
    double shortestDistance = HUGE_VAL;
    for (int landmarkIdx = 0; landmarkIdx < map.landmark_list.size(); ++landmarkIdx) {
      const double distance = dist((double)map.landmark_list[landmarkIdx].x_f,
                                   (double)map.landmark_list[landmarkIdx].y_f,
                                   observations[obsIdx].x, observations[obsIdx].y);
      if (distance < shortestDistance) {
        observations[obsIdx].landmarkIdx = landmarkIdx;
        shortestDistance = distance;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<Observation>& observations,
                                   const Map& map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian
  // distribution. You can read
  //   more about this distribution here:
  //   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your
  // particles are located
  //   according to the MAP'S coordinate system. You will need to transform
  //   between the two systems.
  //   Keep in mind that this transformation requires both rotation AND
  //   translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement
  //   (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to
  // their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
}

Particle ParticleFilter::SetAssociations(Particle& particle,
                                         const std::vector<int>& associations,
                                         const std::vector<double>& sense_x,
                                         const std::vector<double>& sense_y) {
  // particle: the particle to assign each listed association, and association's
  // (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
