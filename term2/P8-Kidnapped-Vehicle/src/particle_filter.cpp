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
  const int num_particles = 1000;

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
  // TODO: Why not use GPS to improve prediction?

  const double eps = 0.0001;
  default_random_engine gen;
  normal_distribution<double> dist_x(0.0, std_pos[0]);
  normal_distribution<double> dist_y(0.0, std_pos[1]);
  normal_distribution<double> dist_theta(0.0, std_pos[2]);

  for (auto particle = particles.begin(); particle != particles.end(); ++particle) {
    /* Add noise to particles to counter prior state uncertainty before
     * predicting the new state using the noiseless motion model (both the
     * velocity and yaw_rate is noiseless according to a comment in main.cpp).
     */
    particle->x += dist_x(gen);
    particle->y += dist_y(gen);
    particle->theta += dist_theta(gen);

    // Prediction using the bicycle motion model
    const double yaw_angle = particle->theta;
    if (fabs(yaw_rate) < eps) {
      // The vehicle is driving straight (this avoids division by zero).
      particle->x += velocity * cos(yaw_angle) * delta_t;
      particle->y += velocity * sin(yaw_angle) * delta_t;
    } else {
      // The vehicle is turning.
      particle->x += (velocity / yaw_rate) * (sin(yaw_angle + (yaw_rate * delta_t)) - sin(yaw_angle));
      particle->y += (velocity / yaw_rate) * (-cos(yaw_angle + (yaw_rate * delta_t)) + cos(yaw_angle));
    }
    particle->theta += yaw_rate * delta_t;
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

void ParticleFilter::updateWeights(double sensor_range, const double std_landmark[],
                                   const std::vector<Observation>& observations,
                                   const Map& map) {
  for (auto particle = particles.begin(); particle != particles.end(); ++particle) {

    // Transform observations to map coordinates. Note that the particle is the observer.
    vector<TransformedObservation> transformedObservations;
    for (auto observation = observations.begin(); observation != observations.end(); ++observation) {
      transformedObservations.push_back(transformObservation(particle->x, particle->y, particle->theta, *observation));
    }

    // Associate each observation with a landmark.
    dataAssociation(map, transformedObservations);

    // Store associations for each particle to provide for visualization in simulator.
    SetAssociations(*particle, transformedObservations, map);

    // Calculate the particle weight as the product of each observation probability.
    particle->weight = 1.0;
    for (auto observation = transformedObservations.begin(); observation != transformedObservations.end(); ++observation) {
      double observationProbability;
      if (observation->landmarkIdx != TransformedObservation::invalidLandmarkIdx) {
        // Calculate the probability that this observation belongs to the associated landmark.
        const Map::single_landmark_s *landmark = &map.landmark_list[observation->landmarkIdx];
        observationProbability = multivariateGaussianProbability(observation->x, observation->y,
                                                                 (double)landmark->x_f, (double)landmark->y_f,
                                                                 std_landmark[0], std_landmark[1]);
      } else {
        // No valid landmark for this observation. Use sensor range to calculate some kind of worst case probability.
        observationProbability = multivariateGaussianProbability(0.0, 0.0,
                                                                 sqrt(sensor_range / 2.0), sqrt(sensor_range / 2.0),
                                                                 std_landmark[0], std_landmark[1]);
      }
      particle->weight *= observationProbability;
    }
  }
}

void ParticleFilter::resample() {
  // Extract weights to vector
  weights.clear();
  for (auto particle = particles.begin(); particle != particles.end(); ++particle) {
    weights.push_back(particle->weight);
  }

  // Create a random distribution based on the particle weights.
  default_random_engine generator;
  discrete_distribution<int> distribution(weights.begin(), weights.end());

  // Draw the required number of particles from the weighted random distribution.
  vector<Particle> resampledParticles;
  for (int i = 0; i < particles.size(); ++i) {
    const int particleIdx = distribution(generator);
    resampledParticles.push_back(particles[particleIdx]);
  }

  // Replace previous particles with the updated distribution.
  particles = resampledParticles;
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<TransformedObservation>& observations,
                                     const Map& map) {
  // Clear any previous association for this particle.
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  // Append each transformed observation along with associated landmark ID.
  for (auto observation = observations.begin(); observation != observations.end(); ++observation) {
    if (observation->landmarkIdx != TransformedObservation::invalidLandmarkIdx) {
      particle.associations.push_back(map.landmark_list[observation->landmarkIdx].id_i);
      particle.sense_x.push_back(observation->x);
      particle.sense_y.push_back(observation->y);
    }
  }
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
