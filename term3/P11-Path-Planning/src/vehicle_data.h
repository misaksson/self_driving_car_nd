#ifndef VEHICLE_DATA_H
#define VEHICLE_DATA_H

#include <iostream>
#include <math.h>
#include <vector>
#include <stdint.h>
#include <tuple>
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
                                                                sqrt(pow(data[3], 2.0) + pow(data[4], 2.0))) {};

    uint64_t id;
    double vx;
    double vy;

    friend std::ostream& operator<<(std::ostream &os, const OtherVehicleData &m) {
      return os << "id=" << m.id <<
                    " " << static_cast <const EgoVehicleData &>(m) <<
                    ", vx=" << m.vx <<
                    ", vy=" << m.vy;
    }
  };

  VehicleData(double ego_x, double ego_y, double ego_s, double ego_d, double ego_yaw, double ego_speed,
              std::vector<std::vector<double>> sensorFusion) {
    ego = EgoVehicleData(ego_x, ego_y, ego_s, ego_d, ego_yaw, ego_speed);
    for (auto otherVehicleData = sensorFusion.begin(); otherVehicleData != sensorFusion.end(); ++otherVehicleData) {
      others.push_back(OtherVehicleData(*otherVehicleData));
    }
  };
  virtual ~VehicleData() {};

  EgoVehicleData ego;
  std::vector<OtherVehicleData> others;
};

#endif /* VEHICLE_DATA_H */
