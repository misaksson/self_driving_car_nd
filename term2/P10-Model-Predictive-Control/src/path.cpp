#include <assert.h>
#include <cmath>
#include <tuple>
#include "Eigen-3.3/Eigen/Dense"
#include "MPC.h"
#include "path.h"
#include "polynomial.h"

using namespace std;

/** Transform a global coordinate to the local coordinate system of the vehicle
 * at current location.
 * @param globalX Global coordinate to transform.
 * @param globalY Global coordinate to transform.
 * @param vehicleX Global coordinate of the vehicle.
 * @param vehicleY Global coordinate of the vehicle.
 * @param vehiclePhi Global orientation of the vehicle.
 * @output The transformed x and y coordinates.
 * */
inline tuple<double, double> global2LocalTransform(const double globalX, const double globalY,
                                                   const double localOffsetX, const double localOffsetY,
                                                   const double localOffsetPsi) {

  Eigen::Matrix3d local2Global;
  local2Global << cos(localOffsetPsi), -sin(localOffsetPsi), localOffsetX,
                  sin(localOffsetPsi), cos(localOffsetPsi), localOffsetY,
                  0.0, 0.0, 1.0;
  Eigen::Matrix3d global2Local = local2Global.inverse(); // ToDo: what is the direct transform?
  Eigen::Vector3d global(globalX, globalY, 1.0);
  Eigen::Vector3d local = global2Local * global;
  return make_tuple(local[0], local[1]);
}

tuple<Eigen::VectorXd, Eigen::VectorXd> global2LocalTransform(const vector<double> &globalX, const vector<double> &globalY,
                                                              const double localOffsetX, const double localOffsetY,
                                                              const double localOffsetPsi) {
  Eigen::VectorXd localX(globalX.size()), localY(globalX.size());
  for (size_t i = 0; i < globalX.size(); ++i) {
    tie(localX[i], localY[i]) = global2LocalTransform(globalX[i], globalY[i], localOffsetX, localOffsetY, localOffsetPsi);
  }
  return make_tuple(localX, localY);

}

Path::Path(const std::vector<double> &waypointsX, const std::vector<double> &waypointsY, const MPC::State globalState) {
  // Transform waypoints to vehicle local coordinate system.
  Eigen::VectorXd localX, localY;
  tie(localX, localY) = global2LocalTransform(waypointsX, waypointsY, globalState.x, globalState.y, globalState.psi);

  // Fit a polynomial to the waypoints given in vehicle local coordinates
  localCoeffs_ = Polynomial::Fit(localX, localY, 3);

  SetLocalState(globalState);
}

Path::~Path() {}

void Path::SetLocalState(MPC::State globalState) {
  localState_.x = 0.0;
  localState_.y = 0.0;
  localState_.psi = 0.0;
  localState_.v = globalState.v;

  /* Crosstrack error measured as the perpendicular distance from the
   * vehicle direction vector to track.
   * ToDo: should it be the closest distance instead?
   * Note: this could be optimized since y(0)=coeffs[0] */
  localState_.cte = Polynomial::Evaluate(localCoeffs_, 0.0);

  /* Orientation error measured at the point on the track that is
   * perpendicular to the vehicle direction.
   * Note: this could be optimized since y'(0)=coeffs[1] */
  localState_.epsi = atan(Polynomial::Evaluate(Polynomial::Derivative(localCoeffs_), 0.0));
}

Eigen::VectorXd Path::GetLocalCoeffs() const {
  return localCoeffs_;
}

MPC::State Path::GetLocalState() const {
  return localState_;
}

std::tuple<std::vector<double>, std::vector<double>> Path::GetLocalLine() const {
  vector<double> lineX, lineY;
  for (double x = 2.0; x <= 75.0; x += 5.0) {
    lineX.push_back(x);
    lineY.push_back(Polynomial::Evaluate(localCoeffs_, x));
  }
  return make_tuple(lineX, lineY);
}
