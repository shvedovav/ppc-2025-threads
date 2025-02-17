#include "seq/shvedova_v_graham_convex_hull_seq/include/ops_seq.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <cmath>
#include <utility>

namespace shvedova_v_graham_convex_hull_seq {

bool GrahamConvexHullSequential::IsLeftAngle(const std::vector<double>& p1,
                                               const std::vector<double>& p2,
                                               const std::vector<double>& p3) {
  return ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])) < 0;
}

bool GrahamConvexHullSequential::PreProcessingImpl() {
  count_point_ = static_cast<int>(task_data->inputs_count[0] / 2);
  input_.resize(count_point_, std::vector<double>(2, 0));
  auto* tmp_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  for (int i = 0; i < count_point_ * 2; i += 2) {
    input_[i / 2][0] = tmp_ptr[i];
    input_[i / 2][1] = tmp_ptr[i + 1];
  }
  return true;
}

bool GrahamConvexHullSequential::ValidationImpl() {
  return (task_data->inputs.size() == 1 &&
          task_data->inputs_count.size() == 1 &&
          task_data->outputs.size() == 2 &&
          task_data->outputs_count.size() == 2 &&
          (task_data->inputs_count[0] % 2 == 0) &&
          (task_data->inputs_count[0] / 2 > 2) &&
          (task_data->outputs_count[0] == 1) &&
          (task_data->outputs_count[1] >= task_data->inputs_count[0]));
}

bool GrahamConvexHullSequential::RunImpl() {
  SetAnchorPoint();
  SortPointsByPolarAngle();
  if (ArePointsCollinear()) return false;
  ConstructHull();
  return true;
}

bool GrahamConvexHullSequential::PostProcessingImpl() {
  reinterpret_cast<int*>(task_data->outputs[0])[0] = count_point_;
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[1]);
  for (int i = 0; i < count_point_; i++) {
    out_ptr[2 * i] = input_[i][0];
    out_ptr[2 * i + 1] = input_[i][1];
  }
  return true;
}

void GrahamConvexHullSequential::SetAnchorPoint() {
  auto min_it = std::min_element(input_.begin(), input_.end(),
    [](const std::vector<double>& a, const std::vector<double>& b) {
      return (a[1] < b[1]) || ((a[1] == b[1]) && (a[0] > b[0]));
    });
  int ind_min_y = static_cast<int>(min_it - input_.begin());
  std::swap(input_[0], input_[ind_min_y]);
}

void GrahamConvexHullSequential::SortPointsByPolarAngle() {
  std::sort(input_.begin() + 1, input_.end(),
    [this](const std::vector<double>& a, const std::vector<double>& b) {
      return IsLeftAngle(a, input_[0], b);
    });
}

bool GrahamConvexHullSequential::ArePointsCollinear() const {
  if (input_.size() < 3) return true;
  double dx = input_[1][0] - input_[0][0];
  double dy = input_[1][1] - input_[0][1];
  for (size_t i = 2; i < input_.size(); i++) {
    double dx_i = input_[i][0] - input_[0][0];
    double dy_i = input_[i][1] - input_[0][1];
    double cross = dx * dy_i - dy * dx_i;
    if (std::fabs(cross) > 1e-9) return false;
  }
  return true;
}

void GrahamConvexHullSequential::ConstructHull() {
  int k = 1;
  for (int i = 2; i < count_point_; i++) {
    while (k > 0 && IsLeftAngle(input_[k - 1], input_[k], input_[i])) {
      k--;
    }
    std::swap(input_[i], input_[k + 1]);
    k++;
  }
  count_point_ = k + 1;
}

}  // namespace shvedova_v_graham_convex_hull_seq
