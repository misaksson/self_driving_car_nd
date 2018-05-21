#ifndef VEHICLE_DATA_H
#define VEHICLE_DATA_H

#include <iostream>
#include <vector>
#include <stdint.h>

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
  struct OtherVehicleData {
    OtherVehicleData() {};
    OtherVehicleData(std::vector<double> data) : id(static_cast<uint64_t>(data[0])), x(data[1]), y(data[2]),
                                                 vx(data[3]), vy(data[4]), s(data[5]), d(data[6]) {};
    uint64_t id;
    double x;
    double y;
    double vx;
    double vy;
    double s;
    double d;

    friend std::ostream& operator<<(std::ostream &os, const OtherVehicleData &m) {
      return os << "id=" << m.id <<
                 ", x=" << m.x <<
                 ", y=" << m.y <<
                 ", vx=" << m.vx <<
                 ", vy=" << m.vy <<
                 ", s=" << m.s <<
                 ", d=" << m.d;
    }
  };

  VehicleData(double ego_x, double ego_y, double ego_s, double ego_d, double ego_yaw, double ego_speed,
              std::vector<std::vector<double>> sensorFusion) {
    ego = EgoVehicleData(ego_x, ego_y, ego_s, ego_d, ego_yaw, ego_speed);
    for (auto otherVehicleData = sensorFusion.begin(); otherVehicleData != sensorFusion.end(); ++otherVehicleData) {
      others.push_back(OtherVehicleData(*otherVehicleData));
    }
  };
  VehicleData(const EgoVehicleData &ego, const std::vector<OtherVehicleData> others) : ego(ego), others(others) {};

  virtual ~VehicleData() {};


  EgoVehicleData ego;
  std::vector<OtherVehicleData> others;
};

#endif /* VEHICLE_DATA_H */
