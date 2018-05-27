#include <fstream>
#include <math.h>
#include <sstream>
#include <tuple>
#include <vector>
#include "Eigen-3.3/Eigen/Dense"
#include "constants.h"
#include "helpers.h"

using namespace std;

Helpers::Helpers(std::string waypointsMapFile) {
  ifstream in_map_(waypointsMapFile.c_str(), ifstream::in);
  string line;
  while (getline(in_map_, line)) {
    istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }
}

Helpers::~Helpers() {}

// For converting back and forth between radians and degrees.
double Helpers::deg2rad(double x) { return x * M_PI / 180; }
double Helpers::rad2deg(double x) { return x * 180 / M_PI; }
double Helpers::milesPerHour2MetersPerSecond(double x) { return x * 0.44704; }

double Helpers::distance(double x1, double y1, double x2, double y2)
{
  return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}

double Helpers::CalcYaw(double x1, double y1, double x2, double y2)
{
  return atan2(y2 - y1, x2 - x1);
}

int Helpers::GetLane(double d) {
  return static_cast<int>(floor((d / constants.laneWidth)));
}

int Helpers::GetLane(double x, double y, double theta) {
  double s, d;
  tie(s, d) = helpers.getFrenet(x, y, theta);
  return GetLane(d);
}

int Helpers::ClosestWaypoint(double x, double y) const {

  double closestLen = 100000; //large number
  int closestWaypoint = 0;

  for(int i = 0; i < map_waypoints_x.size(); i++)
  {
    double map_x = map_waypoints_x[i];
    double map_y = map_waypoints_y[i];
    double dist = distance(x,y,map_x,map_y);
    if(dist < closestLen)
    {
      closestLen = dist;
      closestWaypoint = i;
    }
  }

  return closestWaypoint;
}

double Helpers::calcLongitudinalDiff(double s1, double s2) {
  double longitudinalDiff;
  if ((s1 < constants.trackLength * 0.25) && (s2 > constants.trackLength * 0.75)) {
    // s1 has wrapped around the track.
    longitudinalDiff = (s1 + constants.trackLength) - s2;
  } else if ((s1 > constants.trackLength * 0.75) && (s2 < constants.trackLength * 0.25)) {
    // s2 has wrapped around the track.
    longitudinalDiff = s1 - (s2 + constants.trackLength);
  } else {
    // No wrap-around to consider.
    longitudinalDiff = s1 - s2;
  }
  return longitudinalDiff;
}


int Helpers::NextWaypoint(double x, double y, double theta) const {

  int closestWaypoint = ClosestWaypoint(x,y);

  double map_x = map_waypoints_x[closestWaypoint];
  double map_y = map_waypoints_y[closestWaypoint];

  double heading = atan2((map_y-y),(map_x-x));

  double angle = fabs(theta-heading);
  angle = min(2 * M_PI - angle, angle);

  if(angle > M_PI / 4) {
    closestWaypoint++;
    if (closestWaypoint == map_waypoints_x.size()) {
      closestWaypoint = 0;
    }
  }

  return closestWaypoint;
}

tuple<double, double> Helpers::getFrenet(double x, double y, double theta) const {
  int next_wp = NextWaypoint(x,y, theta);

  int prev_wp;
  prev_wp = next_wp-1;
  if(next_wp == 0)
  {
    prev_wp  = map_waypoints_x.size()-1;
  }

  double n_x = map_waypoints_x[next_wp]-map_waypoints_x[prev_wp];
  double n_y = map_waypoints_y[next_wp]-map_waypoints_y[prev_wp];
  double x_x = x - map_waypoints_x[prev_wp];
  double x_y = y - map_waypoints_y[prev_wp];

  // find the projection of x onto n
  double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
  double proj_x = proj_norm*n_x;
  double proj_y = proj_norm*n_y;

  double frenet_d = distance(x_x,x_y,proj_x,proj_y);

  //see if d value is positive or negative by comparing it to a center point

  double center_x = 1000-map_waypoints_x[prev_wp];
  double center_y = 2000-map_waypoints_y[prev_wp];
  double centerToPos = distance(center_x,center_y,x_x,x_y);
  double centerToRef = distance(center_x,center_y,proj_x,proj_y);

  if(centerToPos <= centerToRef)
  {
    frenet_d *= -1;
  }

  // calculate s value
  double frenet_s = 0;
  for(int i = 0; i < prev_wp; i++)
  {
    frenet_s += distance(map_waypoints_x[i],map_waypoints_y[i],map_waypoints_x[i+1],map_waypoints_y[i+1]);
  }

  frenet_s += distance(0,0,proj_x,proj_y);

  return make_tuple(frenet_s, frenet_d);

}

tuple<double, double> Helpers::getXY(double s, double d) const {
  int prev_wp = -1;

  while(s > map_waypoints_s[prev_wp+1] && (prev_wp < (int)(map_waypoints_s.size()-1) ))
  {
    prev_wp++;
  }

  int wp2 = (prev_wp+1)%map_waypoints_x.size();

  double heading = atan2((map_waypoints_y[wp2]-map_waypoints_y[prev_wp]),(map_waypoints_x[wp2]-map_waypoints_x[prev_wp]));
  // the x,y,s along the segment
  double seg_s = (s-map_waypoints_s[prev_wp]);

  double seg_x = map_waypoints_x[prev_wp]+seg_s*cos(heading);
  double seg_y = map_waypoints_y[prev_wp]+seg_s*sin(heading);

  double perp_heading = heading - M_PI / 2.0;

  double x = seg_x + d*cos(perp_heading);
  double y = seg_y + d*sin(perp_heading);

  return make_tuple(x, y);
}

tuple<double, double> Helpers::global2LocalTransform(const double global_x, const double global_y,
                                                     const double localOffset_x, const double localOffset_y,
                                                     const double localOffset_psi) {
  Eigen::Matrix3d local2Global;
  local2Global << cos(localOffset_psi), -sin(localOffset_psi), localOffset_x,
                  sin(localOffset_psi), cos(localOffset_psi), localOffset_y,
                  0.0, 0.0, 1.0;
  Eigen::Matrix3d global2Local = local2Global.inverse();
  Eigen::Vector3d global(global_x, global_y, 1.0);
  Eigen::Vector3d local = global2Local * global;
  return make_tuple(local[0], local[1]);
}

tuple<double, double> Helpers::local2GlobalTransform(const double local_x, const double local_y,
                                                     const double localOffset_x, const double localOffset_y,
                                                     const double localOffset_psi) {
  Eigen::Matrix3d local2Global;
  local2Global << cos(localOffset_psi), -sin(localOffset_psi), localOffset_x,
                  sin(localOffset_psi), cos(localOffset_psi), localOffset_y,
                  0.0, 0.0, 1.0;
  Eigen::Vector3d local(local_x, local_y, 1.0);
  Eigen::Vector3d global = local2Global * local;
  return make_tuple(global[0], global[1]);
}
