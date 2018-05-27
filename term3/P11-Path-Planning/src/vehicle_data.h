#ifndef VEHICLE_DATA_H
#define VEHICLE_DATA_H

#include <iostream>
#include <math.h>
#include <vector>
#include <stdint.h>
#include <tuple>
#include "constants.h"
#include "helpers.h"

class VehicleData {
public:
  /** Localization data of ego vehicle. */
  struct EgoVehicleData {
    EgoVehicleData() {};
    EgoVehicleData(double x, double y, double s, double d, double yaw, double speed) :
                   x(x), y(y), s(s), d(d), yaw(yaw), speed(speed) {};
    double x;
    double y;
    double s;
    double d;
    double yaw;
    double speed;

    friend std::ostream& operator<<(std::ostream &os, const EgoVehicleData &m) {
      return os << "x=" << m.x <<
                 ", y=" << m.y <<
                 ", s=" << m.s <<
                 ", d=" << m.d <<
                 ", yaw=" << m.yaw <<
                 ", speed=" << m.speed;
    }
  };

  /** Localization data of other vehicles. */
  struct OtherVehicleData : public EgoVehicleData {
    OtherVehicleData() {};
    OtherVehicleData(std::vector<double> data) : id(static_cast<uint64_t>(data[0])),
                                                 vx(data[3]), vy(data[4]),
                                                 EgoVehicleData(data[1], data[2], data[5], data[6],
                                                                atan2(data[4], data[3]),
                                                                sqrt(pow(data[3], 2.0) + pow(data[4], 2.0))) {

      /* Calculate vs and vd from a number of transformed s and d coordinates at different time points in time.
       * The frenet transform is not stable enough to rely on a single value. */
      const double deltaTime = 0.5;
      const double startTime = -2.0;
      const int numCoords = 9u;
      std::vector<double> transformed_s(numCoords), transformed_d(numCoords);
      for (int i = 0; i < numCoords; ++i) {
        const double time = static_cast<double>(i) * deltaTime + startTime;
        std::tie(transformed_s[i], transformed_d[i]) = helpers.getFrenet(x + vx * time, y + vy * time, yaw);
      }

      vs = 0.0;
      vd = 0.0;
      for (int i = 1u; i < numCoords; ++i) {
        vs += Helpers::calcLongitudinalDiff(transformed_s[i], transformed_s[i - 1]) / deltaTime;
        vd += (transformed_d[i] - transformed_d[i - 1]) / deltaTime;
      }
      vs /= static_cast<double>(numCoords - 1);
      vd /= static_cast<double>(numCoords - 1);

      /* Sanity check. */
      const double frenetSpeed = sqrt(pow(vs, 2.0) + pow(vd, 2.0));
      if ((fabs(vd) > 5.0) || (frenetSpeed < (speed - 1.0)) || (frenetSpeed > (speed + 1.0)) ||
          (d < -1.0) || (d > constants.numLanes * constants.laneWidth + 1.0)) {
        // Seems like the frenet coordinates is broken.
        isFrenetValid = false;
//        std::cout << "Bad frenet: " << *this << ", frenetSpeed=" << frenetSpeed << std::endl;
      } else {
        isFrenetValid = true;
      }


    };

    uint64_t id;
    double vx;
    double vy;
    double vs;
    double vd;
    bool isFrenetValid;

    friend std::ostream& operator<<(std::ostream &os, const OtherVehicleData &m) {
      return os << "id=" << m.id <<
                    " " << static_cast <const EgoVehicleData &>(m) <<
                    ", vx=" << m.vx <<
                    ", vy=" << m.vy <<
                    ", vs=" << m.vs <<
                    ", vd=" << m.vd;
    }
  };

  VehicleData(double ego_x, double ego_y, double ego_s, double ego_d, double ego_yaw, double ego_speed,
              std::vector<std::vector<double>> sensorFusion) {
    ego = EgoVehicleData(ego_x, ego_y, ego_s, ego_d, ego_yaw, ego_speed);
    for (auto detection = sensorFusion.begin(); detection != sensorFusion.end(); ++detection) {
      others.push_back(OtherVehicleData(*detection));
    }
  };
  virtual ~VehicleData() {};

  EgoVehicleData ego;
  std::vector<OtherVehicleData> others;
};

#endif /* VEHICLE_DATA_H */
