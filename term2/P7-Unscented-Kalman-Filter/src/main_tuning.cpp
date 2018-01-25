#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include <vector>
#include "ukf.h"
#include "tools.h"

using namespace std;

vector<string> readDataSet(string filename);
VectorXd processDataSet(vector<string> dataSet);

int main()
{
  VectorXd RMSE1 = processDataSet(readDataSet("../data/obj_pose-laser-radar-synthetic-input.txt"));
  VectorXd RMSE2 = processDataSet(readDataSet("../data/obj_pose-laser-radar-synthetic-input2.txt"));

  cout << "RMSE " << RMSE1.transpose() << " " << RMSE2.transpose() << endl;
}

vector<string> readDataSet(string filename) {
  vector<string> dataSet;
  ifstream dataFile;
  dataFile.open(filename);
  if (dataFile.is_open()) {
    string line;
    while (getline(dataFile, line)) {
      dataSet.push_back(line);
    }
    dataFile.close();
  } else {
    cout << "Unable to open file " << filename << endl;
  }
  return dataSet;
}

VectorXd processDataSet(vector<string> dataSet) {
 // Create a Kalman Filter instance
  UKF ukf;

  // used to compute the RMSE later
  vector<VectorXd> estimations;
  vector<VectorXd> ground_truth;

  MeasurementPackage meas_package;

  for (auto sensorMeasurment = dataSet.begin(); sensorMeasurment != dataSet.end(); ++sensorMeasurment) {
    istringstream iss(*sensorMeasurment);
    long long timestamp;

    // reads first element from the current line
    string sensor_type;
    iss >> sensor_type;

    if (sensor_type.compare("L") == 0) {
      meas_package.sensor_type_ = MeasurementPackage::LASER;
      meas_package.raw_measurements_ = VectorXd(2);
      float px;
      float py;
      iss >> px;
      iss >> py;
      meas_package.raw_measurements_ << px, py;
      iss >> timestamp;
      meas_package.timestamp_ = timestamp;
    } else if (sensor_type.compare("R") == 0) {
      meas_package.sensor_type_ = MeasurementPackage::RADAR;
      meas_package.raw_measurements_ = VectorXd(3);
      float ro;
      float theta;
      float ro_dot;
      iss >> ro;
      iss >> theta;
      iss >> ro_dot;
      meas_package.raw_measurements_ << ro,theta, ro_dot;
      iss >> timestamp;
      meas_package.timestamp_ = timestamp;
    }
    float x_gt;
    float y_gt;
    float vx_gt;
    float vy_gt;
    iss >> x_gt;
    iss >> y_gt;
    iss >> vx_gt;
    iss >> vy_gt;
    VectorXd gt_values(4);
    gt_values(0) = x_gt;
    gt_values(1) = y_gt;
    gt_values(2) = vx_gt;
    gt_values(3) = vy_gt;
    ground_truth.push_back(gt_values);

      //Call ProcessMeasurment(meas_package) for Kalman filter
    ukf.ProcessMeasurement(meas_package);

    //Push the current estimated x,y positon from the Kalman filter's state vector

    VectorXd estimate(4);

    double p_x = ukf.x_(0);
    double p_y = ukf.x_(1);
    double v  = ukf.x_(2);
    double yaw = ukf.x_(3);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    estimate(0) = p_x;
    estimate(1) = p_y;
    estimate(2) = v1;
    estimate(3) = v2;

    estimations.push_back(estimate);
  }
  VectorXd RMSE = Tools::CalculateRMSE(estimations, ground_truth);
  return RMSE;
}
