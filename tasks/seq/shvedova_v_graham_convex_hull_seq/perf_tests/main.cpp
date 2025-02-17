#include <gtest/gtest.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <random>
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/shvedova_v_graham_convex_hull_seq/include/ops_seq.hpp"

TEST(shvedova_v_graham_convex_hull_seq, test_pipeline_run) {
  constexpr int kCount = 900000;
  std::vector<double> points;
  points.reserve(kCount * 2);
  std::mt19937 gen(42);
  std::uniform_real_distribution<> dis(-100.0, 100.0);
  for (int i = 0; i < kCount; ++i) {
    points.push_back(dis(gen));
    points.push_back(dis(gen));
  }
  int scan_size = 0;
  std::vector<double> hull(points.size(), 0.0);
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(points.data()));
  task_data_seq->inputs_count.emplace_back(points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&scan_size));
  task_data_seq->outputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(hull.data()));
  task_data_seq->outputs_count.emplace_back(hull.size());
  auto test_task_sequential = std::make_shared<shvedova_v_graham_convex_hull_seq::GrahamConvexHullSequential>(task_data_seq);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  EXPECT_GE(static_cast<size_t>(scan_size), 3U);
}

TEST(shvedova_v_graham_convex_hull_seq, test_task_run) {
  constexpr int kCount = 900000;
  std::vector<double> points;
  points.reserve(kCount * 2);
  std::mt19937 gen(42);
  std::uniform_real_distribution<> dis(-100.0, 100.0);
  for (int i = 0; i < kCount; ++i) {
    points.push_back(dis(gen));
    points.push_back(dis(gen));
  }
  int scan_size = 0;
  std::vector<double> hull(points.size(), 0.0);
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(points.data()));
  task_data_seq->inputs_count.emplace_back(points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&scan_size));
  task_data_seq->outputs_count.emplace_back(1);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(hull.data()));
  task_data_seq->outputs_count.emplace_back(hull.size());
  auto test_task_sequential = std::make_shared<shvedova_v_graham_convex_hull_seq::GrahamConvexHullSequential>(task_data_seq);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  EXPECT_GE(static_cast<size_t>(scan_size), 3U);
}
