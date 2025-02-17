#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include "core/task/include/task.hpp"

namespace shvedova_v_graham_convex_hull_seq {

class GrahamConvexHullSequential : public ppc::core::Task {
 public:
  explicit GrahamConvexHullSequential(ppc::core::TaskDataPtr task_data)
      : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int count_point_{0};
  std::vector<std::vector<double>> input_;

  static bool IsLeftAngle(const std::vector<double>& p1,
                          const std::vector<double>& p2,
                          const std::vector<double>& p3);

  void SetAnchorPoint();
  void SortPointsByPolarAngle();
  void ConstructHull();
  bool ArePointsCollinear() const;
};

}  // namespace shvedova_v_graham_convex_hull_seq
