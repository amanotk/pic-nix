// -*- C++ -*-
#include "balancer.hpp"

#define DEFINE_MEMBER(type, name) type Balancer::name

NIX_NAMESPACE_BEGIN

DEFINE_MEMBER(bool, assign_smilei)(std::vector<float64>& load, std::vector<int>& boundary)
{
  const int Nc = load.size();
  const int Nr = boundary.size() - 1;

  float64              mean_load = 0;
  std::vector<float64> cumload(Nc + 1);
  std::vector<int>     old_boundary(Nr + 1);

  // calculate cumulative load
  cumload[0] = 0;
  for (int i = 0; i < Nc; i++) {
    cumload[i + 1] = cumload[i] + load[i];
  }
  mean_load = cumload[Nc] / Nr;

  // copy original boundary
  std::copy(boundary.begin(), boundary.end(), old_boundary.begin());

  // process for each rank boundary
  for (int i = 1; i < Nr; i++) {
    float64 target  = mean_load * i;
    float64 current = cumload[boundary[i]];

    if (current > target) {
      //
      // possibly move boundary backward
      //
      int index = boundary[i] - 1;

      while (std::abs(current - target) > std::abs(current - target - load[index])) {
        current -= load[index];
        index--;
      }

      // set new boundary
      if (index >= old_boundary[i - 1]) {
        boundary[i] = index + 1;
      } else {
        boundary[i] = old_boundary[i - 1] + 1; // accommodate at least one chunk
      }
    } else {
      //
      // move boundary forward
      //
      int index = boundary[i];

      while (std::abs(current - target) > std::abs(current - target + load[index])) {
        current += load[index];
        index++;
      }

      // set new boundary
      if (index < old_boundary[i + 1]) {
        boundary[i] = index;
      } else {
        boundary[i] = old_boundary[i + 1] - 1; // accommodate at least one chunk
      }
    }
  }

  bool is_updated = std::equal(boundary.begin(), boundary.end(), old_boundary.begin()) == false;

  return is_updated;
}

DEFINE_MEMBER(bool, assign_binarysearch)(std::vector<float64>& load, std::vector<int>& boundary)
{
  const int Nc = load.size();
  const int Nr = boundary.size() - 1;

  float64              mean_load = 0;
  std::vector<float64> cumload(Nc + 1);

  // calculate cumulative sum
  cumload[0] = 0;
  for (int i = 0; i < Nc; i++) {
    cumload[i + 1] = cumload[i] + load[i];
  }
  mean_load = cumload[Nc] / Nr;

  boundary[0]  = 0;
  boundary[Nr] = Nc;

  for (int i = 1; i < Nr; i++) {
    auto it     = std::upper_bound(cumload.begin(), cumload.end(), mean_load * i);
    int  index  = std::distance(cumload.begin(), it) - 1;
    boundary[i] = index;
  }

  return is_boundary_ascending(boundary);
}

DEFINE_MEMBER(std::vector<int>, assign_initial)
(int nprocess)
{
  std::vector<int> boundary(nprocess + 1);

  // try to find initial best assignment via binary search
  bool status = assign_binarysearch(chunkload, boundary);

  // if failed, use iterative method
  if (status == false) {
    // uniform load with the same size as load for initialization
    std::vector<float64> uniform_load(nchunk, 1.0);
    assign_binarysearch(uniform_load, boundary);

    // iteratively find best assignment
    static constexpr int maxiter = 100;
    for (int i = 0; i < maxiter; i++) {
      if (assign_smilei(chunkload, boundary) == false)
        break;
    }
  }

  return boundary;
}

DEFINE_MEMBER(std::vector<int>, assign)
(std::vector<int>& boundary)
{
  assign_smilei(chunkload, boundary);

  return boundary;
}

DEFINE_MEMBER(std::vector<float64>, get_rankload)
(std::vector<int>& boundary, std::vector<float64>& load)
{
  const int Nr = boundary.size() - 1;

  std::vector<float64> rankload(Nr);
  for (int r = 0; r < Nr; r++) {
    rankload[r] = 0.0;
    for (int i = boundary[r]; i < boundary[r + 1]; i++) {
      rankload[r] += load[i];
    }
  }

  return rankload;
}

DEFINE_MEMBER(void, print_assignment)
(std::ostream& out, std::vector<int>& boundary)
{
  const int Nr = boundary.size() - 1;

  std::vector<float64> rankload = get_rankload(boundary, chunkload);
  float64              meanload = std::accumulate(chunkload.begin(), chunkload.end(), 0.0) / Nr;

  tfm::format(out, "*** mean load = %12.5e ***\n", meanload);
  for (int i = 0; i < Nr; i++) {
    int     numchunk  = boundary[i + 1] - boundary[i];
    float64 deviation = (rankload[i] - meanload) / meanload * 100;
    tfm::format(out, "load[%4d] = %12.5e (%4d : %+7.2f %%)\n", i, rankload[i], numchunk, deviation);
  }
}

DEFINE_MEMBER(bool, is_boundary_ascending)(const std::vector<int>& boundary)
{
  const int nprocess = boundary.size() - 1;

  bool status = true;

  status = status & (boundary[0] == 0);
  status = status & (boundary[nprocess] == nchunk);

  for (int i = 1; i < nprocess; i++) {
    status = status & (boundary[i + 1] > boundary[i]);
  }

  return status;
}

DEFINE_MEMBER(bool, is_boundary_optimum)(const std::vector<int>& boundary)
{
  const int nprocess = boundary.size() - 1;

  bool status = true;

  std::vector<float64> cumulative_load(nchunk + 1, 0.0);
  std::partial_sum(chunkload.begin(), chunkload.end(), cumulative_load.begin() + 1);

  for (int i = 1; i < nprocess; i++) {
    int     index1   = boundary[i];
    int     index2   = boundary[i] + 1;
    float64 bestload = i * cumulative_load[nchunk] / nprocess;

    status = status & (cumulative_load[index1] <= bestload);
    status = status & (cumulative_load[index2] > bestload);
  }

  return status;
}

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
