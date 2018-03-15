#ifndef PATH_H
#define PATH_H

#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"

class Path {
public:
  /** Calculates target path in vehicle local coordinates-
   * @param waypointsX Global x coordinates of the path waypoints.
   * @param waypointsX Global y coordinates of the path waypoints.
   * @param globalState State vector of the vehicle in global coordinates.
   */
  Path(const std::vector<double> &waypointsX, const std::vector<double> &waypointsY, const MPC::State globalState);
  virtual ~Path();

  /** Get polynomial coefficients representing the expected road path.
   * The polynomial is in vehicle local coordinates.
   * @output Coefficients of a third degree polynomial.
   */
  Eigen::VectorXd GetLocalCoeffs() const;
  /** Get vehicle state in vehicle local coordinates.
   * @output Vehicle state in local coordinates.
   */
  MPC::State GetLocalState() const;
  /** Get points of a line representing the expected road path.
   * The points are in vehicle local coordinates.
   * @output Two vectors, containing x- resp. y-coordinates of the line.
   */
  std::tuple<std::vector<double>, std::vector<double>> GetLocalLine() const;

private:
  void SetLocalState(MPC::State globalState);

  Eigen::VectorXd localCoeffs_;
  MPC::State localState_;
};

#endif /* PATH_H */
