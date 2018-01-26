#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include <vector>
#include "ukf.h"
#include "tools.h"

using namespace std;

vector<string> readDataSet(string filename);
void processDataSet(vector<string> dataSet, VectorXd &RMSE,
                    vector<double> &NIS_radar, vector<double> &NIS_lidar,
                    const double std_a = 0.46, const double std_yawdd = 0.54);
void vector2file(vector<double> v, string filename);
string analyseNis(const vector<double> nis, const double chi95);

int main()
{
  VectorXd RMSE;
  vector<double> NIS_radar, NIS_lidar;
  vector<string> dataSet;

  // Dataset 1
  cout << endl << "Data set 1" << endl;
  dataSet = readDataSet("../data/obj_pose-laser-radar-synthetic-input.txt");
  processDataSet(dataSet, RMSE, NIS_radar, NIS_lidar);
  cout << "RMSE:" << RMSE.transpose() << endl;
  cout << "Radar: " << analyseNis(NIS_radar, 7.815) << endl;
  cout << "Lidar: " << analyseNis(NIS_lidar, 5.991) << endl;
  vector2file(NIS_radar, "../NIS1_radar.txt");
  vector2file(NIS_lidar, "../NIS1_lidar.txt");

  // Dataset 2
  cout << endl << "Data set 2" << endl;
  dataSet = readDataSet("../data/obj_pose-laser-radar-synthetic-input2.txt");
  processDataSet(dataSet, RMSE, NIS_radar, NIS_lidar);
  cout << "RMSE:" << RMSE.transpose() << endl;
  cout << "Radar: " << analyseNis(NIS_radar, 7.815) << endl;
  cout << "Lidar: " << analyseNis(NIS_lidar, 5.991) << endl;
  vector2file(NIS_radar, "../NIS2_radar.txt");
  vector2file(NIS_lidar, "../NIS2_lidar.txt");
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

void processDataSet(vector<string> dataSet, VectorXd &RMSE,
                    vector<double> &NIS_radar, vector<double> &NIS_lidar,
                    const double std_a, const double std_yawdd) {

  NIS_radar.clear();
  NIS_lidar.clear();

  // Create a Kalman Filter instance
  UKF ukf(std_a, std_yawdd);

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

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      NIS_radar.push_back(ukf.NIS_value_);
    } else {
      NIS_lidar.push_back(ukf.NIS_value_);
    }
  }
  RMSE = Tools::CalculateRMSE(estimations, ground_truth);
}


void vector2file(vector<double> v, string filename) {
  ofstream fs;
  fs.open(filename);
  if (fs.is_open()) {
    for (auto it = v.begin(); it != v.end(); ++it) {
      fs << *it << endl;
    }
    fs.close();
  } else {
    cout << "Unable to open file " << filename << endl;
  }
}

string analyseNis(const vector<double> nis, const double chi95) {
  int countBelow = 0;
  for (auto it = nis.begin(); it != nis.end(); ++it) {
    countBelow += *it < chi95 ? 1 : 0;
  }
  ostringstream result;
  result << "Amount below chi_squared 95% (" << chi95 << "): " << (double)countBelow / (double)nis.size();
  return result.str();
}
