#include <math.h>
#include <uWS/uWS.h>
#include <iostream>
#include "json.hpp"
#include "ukf.h"

using namespace std;

// for convenience
using json = nlohmann::json;

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
std::string hasData(std::string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("]");
  if (found_null != std::string::npos) {
    return "";
  } else if (b1 != std::string::npos && b2 != std::string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}


/**
 * Predict position at delta_t by assuming constant velocity and yaw
 * rate. */
VectorXd simplePredict(const VectorXd &x, const double delta_t) {
  const double px0 = x(0);
  const double py0 = x(1);
  const double v = x(2);
  const double yaw = x(3);
  const double yawd = x(4);
  const double eps = 0.0001;
  double px1;
  double py1;

  // Avoid division by zero
  if (fabs(yawd) < eps) {
    // Target is driving straight at constant velocity.
    px1 = v * cos(yaw) * delta_t + px0;
    py1 = v * sin(yaw) * delta_t + py0;
  } else {
    // Target is turning at constant yaw rate and velocity.
    // px1 = v0 * integral(cos(yaw0 + yawd0 * (t - t0), dt) + px0 =
    // http://www.wolframalpha.com/input/?i=v+int+cos(a+%2B+b+*+(t+-+c+))+dt,++t+%3D+c+to+d
    px1 = (v / yawd) * (sin(yaw + yawd * delta_t) - sin(yaw)) + px0;
    // py1 = v0 * integral(sin(yaw0 + yawd0 * (t - t0), dt) + py0 =
    // http://www.wolframalpha.com/input/?i=v+int+sin(a+%2B+b+*+(t+-+c+))+dt,++t+%3D+c+to+d
    py1 = (v / yawd) * (-cos(yaw + yawd * delta_t) + cos(yaw)) + py0;
  }

  VectorXd result = VectorXd(2);
  result << px1, py1;
  return result;
}

/**
 * Calculate the distance between position v1 and v2.
 */
double calcDistance(const VectorXd &v1, const VectorXd &v2) {
  return sqrt(pow(v1(0) - v2(0), 2.0) + pow(v1(1) - v2(1), 2.0));
}


/**
 * Find position where the hunter should be able to intercept the target.
 * @return The estimated intercept position.
 */
VectorXd findInterceptPosition(const VectorXd &hunter, const VectorXd &target) {
  /**
   * Trial n' error iterative solution. The prediction time is increased until
   * there is a straight path that should intercept the vehicle.
   */

  // Prediction steps delta time
  const double delta_t = 0.01;

  /* Never try to predict more than this number of second ahead of time. This
   * helps avoid corner cases that causes hang ups in the prediction loop,
   * e.g. when next iteration is just slightly better, which typically happens
   * before the Kalman filter state has stabilized.
   */
  const double maxPredictionTime = 5.0;

  /* The speed of the hunter is unknown, but we know that it's the same as the
   * target so lets use that estimate.
   */
  const double velocity = target(2);

  // Initial state
  VectorXd prediction = target;
  double predictionTime = 0.0;
  double timeToTarget = calcDistance(hunter, prediction) / velocity;
  VectorXd prevPrediction;
  double prevTimeToTarget;
  do {
    // Store previous prediction
    prevPrediction = prediction;
    prevTimeToTarget = timeToTarget;

    // Increment predictionTime until the hunter can intercept the target
    predictionTime += delta_t;

    // Predict target position at predictionTime
    prediction = simplePredict(target, predictionTime);

    // Calculate how long time the hunter needs to reach predicted target position.
    timeToTarget = calcDistance(hunter, prediction) / velocity;

    // Continue until the prediction becomes worse, or until reaching maxPredictionTime.
  } while(abs(timeToTarget - predictionTime) < abs(prevTimeToTarget - (predictionTime - delta_t)) &&
          predictionTime < maxPredictionTime);
  return prevPrediction;
}


int main() {
  uWS::Hub h;

  // Create a UKF instance.
  UKF ukf;

  h.onMessage([&ukf](uWS::WebSocket<uWS::SERVER> ws,
                                           char *data, size_t length,
                                           uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event

    if (length && length > 2 && data[0] == '4' && data[1] == '2') {
      auto s = hasData(std::string(data));
      if (s != "") {
        auto j = json::parse(s);
        std::string event = j[0].get<std::string>();

        if (event == "telemetry") {
          // j[1] is the data JSON object
          VectorXd hunter = VectorXd(2);
          hunter(0) = std::stod(j[1]["hunter_x"].get<std::string>());
          hunter(1) = std::stod(j[1]["hunter_y"].get<std::string>());
          double hunter_heading =
              std::stod(j[1]["hunter_heading"].get<std::string>());

          string lidar_measurment = j[1]["lidar_measurement"];

          MeasurementPackage meas_package_L;
          istringstream iss_L(lidar_measurment);
          long long timestamp_L;

          // reads first element from the current line
          string sensor_type_L;
          iss_L >> sensor_type_L;

          // read measurements at this timestamp
          meas_package_L.sensor_type_ = MeasurementPackage::LASER;
          meas_package_L.raw_measurements_ = VectorXd(2);
          float px;
          float py;
          iss_L >> px;
          iss_L >> py;
          meas_package_L.raw_measurements_ << px, py;
          iss_L >> timestamp_L;
          meas_package_L.timestamp_ = timestamp_L;

          ukf.ProcessMeasurement(meas_package_L);

          string radar_measurment = j[1]["radar_measurement"];

          MeasurementPackage meas_package_R;
          istringstream iss_R(radar_measurment);
          long long timestamp_R;

          // reads first element from the current line
          string sensor_type_R;
          iss_R >> sensor_type_R;

          // read measurements at this timestamp
          meas_package_R.sensor_type_ = MeasurementPackage::RADAR;
          meas_package_R.raw_measurements_ = VectorXd(3);
          float ro;
          float theta;
          float ro_dot;
          iss_R >> ro;
          iss_R >> theta;
          iss_R >> ro_dot;
          meas_package_R.raw_measurements_ << ro, theta, ro_dot;
          iss_R >> timestamp_R;
          meas_package_R.timestamp_ = timestamp_R;

          ukf.ProcessMeasurement(meas_package_R);
          const VectorXd interceptPosition = findInterceptPosition(hunter, ukf.x_);

          double heading_to_target =
              atan2(interceptPosition(1) - hunter(1), interceptPosition(0) - hunter(0));
          while (heading_to_target > M_PI) heading_to_target -= 2. * M_PI;
          while (heading_to_target < -M_PI) heading_to_target += 2. * M_PI;
          // turn towards the target
          double heading_difference = heading_to_target - hunter_heading;
          while (heading_difference > M_PI) heading_difference -= 2. * M_PI;
          while (heading_difference < -M_PI) heading_difference += 2. * M_PI;

          double distance_difference =
              sqrt((ukf.x_[1] - hunter(1)) * (ukf.x_[1] - hunter(1)) +
                   (ukf.x_[0] - hunter(0)) * (ukf.x_[0] - hunter(0)));

          json msgJson;
          msgJson["turn"] = heading_difference;
          msgJson["dist"] = distance_difference;
          auto msg = "42[\"move_hunter\"," + msgJson.dump() + "]";
          // std::cout << msg << std::endl;
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }

  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
  return 0;
}
