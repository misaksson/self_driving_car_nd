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
                    vector<VectorXd> &estimations, vector<VectorXd> &ground_truth,
                    vector<VectorXd> &radar, vector<VectorXd> &lidar,
                    vector<double> &NIS_radar, vector<double> &NIS_lidar,
                    const double std_a = 0.46, const double std_yawdd = 0.54);
void nis2file(vector<double> v, string filename);
void coords2file(const vector<VectorXd> coords, const string filename);
string analyseNis(const vector<double> nis, const double chi95);

int main()
{
  VectorXd RMSE;
  vector<double> NIS_radar, NIS_lidar;
  vector<VectorXd> estimations, ground_truth;
  vector<VectorXd> radar, lidar;
  vector<string> dataSet;

  // Dataset 1
  cout << endl << "Data set 1" << endl;
  dataSet = readDataSet("../data/obj_pose-laser-radar-synthetic-input.txt");
  processDataSet(dataSet, RMSE, estimations, ground_truth, radar, lidar, NIS_radar, NIS_lidar);
  cout << "RMSE:" << RMSE.transpose() << endl;
  cout << "Radar: " << analyseNis(NIS_radar, 7.815) << endl;
  cout << "Lidar: " << analyseNis(NIS_lidar, 5.991) << endl;
  nis2file(NIS_radar, "../NIS1_radar.txt");
  nis2file(NIS_lidar, "../NIS1_lidar.txt");
  coords2file(estimations, "../estimations1.txt");
  coords2file(ground_truth, "../ground_truth1.txt");
  coords2file(radar, "../radar1.txt");
  coords2file(lidar, "../lidar1.txt");

  // Dataset 2
  cout << endl << "Data set 2" << endl;
  dataSet = readDataSet("../data/obj_pose-laser-radar-synthetic-input2.txt");
  processDataSet(dataSet, RMSE, estimations, ground_truth, radar, lidar, NIS_radar, NIS_lidar);
  cout << "RMSE:" << RMSE.transpose() << endl;
  cout << "Radar: " << analyseNis(NIS_radar, 7.815) << endl;
  cout << "Lidar: " << analyseNis(NIS_lidar, 5.991) << endl;
  nis2file(NIS_radar, "../NIS2_radar.txt");
  nis2file(NIS_lidar, "../NIS2_lidar.txt");
  coords2file(estimations, "../estimations2.txt");
  coords2file(ground_truth, "../ground_truth2.txt");
  coords2file(radar, "../radar2.txt");
  coords2file(lidar, "../lidar2.txt");
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
                    vector<VectorXd> &estimations, vector<VectorXd> &ground_truth,
                    vector<VectorXd> &radar, vector<VectorXd> &lidar,
                    vector<double> &NIS_radar, vector<double> &NIS_lidar,
                    const double std_a, const double std_yawdd) {

  // Clear output before starting
  estimations.clear();
  ground_truth.clear();
  radar.clear();
  lidar.clear();
  NIS_radar.clear();
  NIS_lidar.clear();

  // Create a Kalman Filter instance
  UKF ukf(std_a, std_yawdd);

  for (auto sensorMeasurment = dataSet.begin(); sensorMeasurment != dataSet.end(); ++sensorMeasurment) {
    istringstream iss(*sensorMeasurment);
    MeasurementPackage meas_package;
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
      lidar.push_back(meas_package.raw_measurements_);
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
      radar.push_back(Tools::PolarToCartesian(meas_package.raw_measurements_));
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


void nis2file(vector<double> v, string filename) {
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

void coords2file(const vector<VectorXd> coords, const string filename) {
  ofstream fs;
  fs.open(filename);
  if (fs.is_open()) {
    for (auto it = coords.begin(); it != coords.end(); ++it) {
      for (int i = 0; i < it->size(); ++i) {
        fs << (*it)[i] << " ";
      }
      fs << endl;
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
