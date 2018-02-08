#include "catch.hpp"
#include "../src/particle_filter.h"
#include <iostream>
#include <cmath>

using namespace std;

TEST_CASE("Particle filter initialized according to GPS position", "[init]") {
  const double xGps = 1.0, yGps = 2.0, thetaGps = 3.0;
  const double xGpsStd = 0.1, yGpsStd = 0.2, thetaGpsStd = 0.3;
  ParticleFilter pf;
  double stdevGps[] = {xGpsStd, yGpsStd, thetaGpsStd};

  pf.init(xGps, yGps, thetaGps, stdevGps);

  // Calculate mean values.
  double xSum = 0.0, ySum = 0.0, thetaSum = 0.0;
  for (int i = 0; i < pf.particles.size(); ++i) {
    xSum += pf.particles[i].x;
    ySum += pf.particles[i].y;
    thetaSum += pf.particles[i].theta;
  }
  const double xMean = xSum / (double)pf.particles.size();
  const double yMean = ySum / (double)pf.particles.size();
  const double thetaMean = thetaSum / (double)pf.particles.size();

  // Verify that mean values are roughly within expectations.
  REQUIRE(xMean == Approx(xGps).margin(xGpsStd * 0.1));
  REQUIRE(yMean == Approx(yGps).margin(yGpsStd * 0.1));
  REQUIRE(thetaMean == Approx(thetaGps).margin(thetaGpsStd * 0.1));

  // Calculate standard deviations
  double xSqDevSum = 0.0, ySqDevSum = 0.0, thetaSqDevSum = 0.0;
  for (int i = 0; i < pf.particles.size(); ++i) {
    xSqDevSum += pow(pf.particles[i].x - xMean, 2.0);
    ySqDevSum += pow(pf.particles[i].y - yMean, 2.0);
    thetaSqDevSum += pow(pf.particles[i].theta - thetaMean, 2.0);
  }
  const double xStdev = sqrt(xSqDevSum / (double)pf.particles.size());
  const double yStdev = sqrt(ySqDevSum / (double)pf.particles.size());
  const double thetaStdev = sqrt(thetaSqDevSum / (double)pf.particles.size());

  // Verify that standard deviations are roughly within expectations.
  REQUIRE(xStdev == Approx(xGpsStd).margin(xGpsStd * 0.1));
  REQUIRE(yStdev == Approx(yGpsStd).margin(yGpsStd * 0.1));
  REQUIRE(thetaStdev == Approx(thetaGpsStd).margin(thetaGpsStd * 0.1));
}
