#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <tuple>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "path.h"
#include "polynomial.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;
using namespace std;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }
double seconds2milliseconds(double x) { return x * 1000.0; }
double milesPerHour2MetersPerSecond(double x) { return x * 0.44704; }

const double latency = 0.1;

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc(latency);
  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    //cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px = static_cast<double>(j[1]["x"]);
          double py = static_cast<double>(j[1]["y"]);
          double psi = static_cast<double>(j[1]["psi"]);
          double v = milesPerHour2MetersPerSecond(static_cast<double>(j[1]["speed"]));
          MPC::Actuations currentActuations;
          currentActuations.delta = -static_cast<double>(j[1]["steering_angle"]);
          currentActuations.a = static_cast<double>(j[1]["throttle"]);

          // Current state vector in global coordinates.
          MPC::State currentState = {.x = px, .y = py, .psi = psi, .v = v};
          // Predicted state vector in global coordinates.
          MPC::State predictedState = mpc.Predict(currentState, currentActuations);

          // Get path information in vehicle local coordinate system.
          Path path(ptsx, ptsy, predictedState);
          Eigen::VectorXd localCoeffs = path.GetLocalCoeffs();
          MPC::State localState = path.GetLocalState();

          /* Get a line representing the target track. This will be displayed
           * in yellow by the simulator. */
          vector<double> trackX, trackY;
          tie(trackX, trackY) = path.GetLocalLine();

          /* MPC calculates next actuations to apply. */
          MPC::Actuations nextActuations;
          /* The predicted vehicle trajectory will also be provided by the MPC
          * solver. This line is displayed in green by the simulator. */
          vector<double> trajectoryX, trajectoryY;
          tie(nextActuations, trajectoryX, trajectoryY) = mpc.Solve(localState, localCoeffs);

          json msgJson;
          msgJson["steering_angle"] = -nextActuations.delta / deg2rad(25); // Map to range [-1..1]
          msgJson["throttle"] = nextActuations.a;
          msgJson["mpc_x"] = trajectoryX;
          msgJson["mpc_y"] = trajectoryY;
          msgJson["next_x"] = trackX;
          msgJson["next_y"] = trackY;

          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          //std::cout << msg << std::endl;

          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(static_cast<int>(seconds2milliseconds(latency))));
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
}
