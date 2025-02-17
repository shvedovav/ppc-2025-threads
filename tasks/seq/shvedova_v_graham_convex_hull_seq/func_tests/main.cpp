#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <random>
#include <cstddef>
#include <cstdint>
#include "seq/shvedova_v_graham_convex_hull_seq/include/ops_seq.hpp"

namespace {
std::shared_ptr<ppc::core::TaskData> buildTaskData(const std::vector<double>& src,
                                                   std::vector<double>& dst,
                                                   int& hullCount) {
  auto data = std::make_shared<ppc::core::TaskData>();
  data->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(src.data())));
  data->inputs_count.emplace_back(src.size());
  data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&hullCount));
  data->outputs_count.emplace_back(1);
  data->outputs.emplace_back(reinterpret_cast<uint8_t*>(dst.data()));
  data->outputs_count.emplace_back(dst.size());
  return data;
}

void executeTask(const std::shared_ptr<ppc::core::TaskData>& data) {
  shvedova_v_graham_convex_hull_seq::GrahamConvexHullSequential task(data);
  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());
}
}  // namespace

TEST(shvedova_v_graham_convex_hull_seq, convex_triangle) {
  std::vector<double> src = {0.0, 0.0, 2.0, 2.0, 2.0, 0.0};
  int hullCount = 0;
  std::vector<double> dst(src.size(), 0.0);
  auto data = buildTaskData(src, dst, hullCount);
  executeTask(data);
  std::vector<double> exp = {2.0, 0.0, 2.0, 2.0, 0.0, 0.0};
  EXPECT_EQ(static_cast<size_t>(hullCount), exp.size() / 2);
  EXPECT_EQ(std::vector<double>(dst.begin(), dst.begin() + exp.size()), exp);
}

TEST(shvedova_v_graham_convex_hull_seq, convex_square_inner) {
  std::vector<double> src = {0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 1.0, 1.0};
  int hullCount = 0;
  std::vector<double> dst(src.size(), 0.0);
  auto data = buildTaskData(src, dst, hullCount);
  executeTask(data);
  // Ожидается, что внутренняя точка исключена, поэтому оболочка состоит из 4 точек
  std::vector<double> exp = {2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 0.0, 0.0};
  EXPECT_EQ(static_cast<size_t>(hullCount), exp.size() / 2);
  EXPECT_EQ(std::vector<double>(dst.begin(), dst.begin() + exp.size()), exp);
}

TEST(shvedova_v_graham_convex_hull_seq, convex_identical_points) {
  std::vector<double> src = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  int hullCount = 0;
  std::vector<double> dst(src.size(), 0.0);
  auto data = buildTaskData(src, dst, hullCount);
  shvedova_v_graham_convex_hull_seq::GrahamConvexHullSequential task(data);
  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  EXPECT_FALSE(task.RunImpl());
}

TEST(shvedova_v_graham_convex_hull_seq, convex_collinear_diagonal) {
  std::vector<double> src = {0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0};
  int hullCount = 0;
  std::vector<double> dst(src.size(), 0.0);
  auto data = buildTaskData(src, dst, hullCount);
  shvedova_v_graham_convex_hull_seq::GrahamConvexHullSequential task(data);
  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  EXPECT_FALSE(task.RunImpl());
}

TEST(shvedova_v_graham_convex_hull_seq, convex_collinear_vertical) {
  std::vector<double> src = {0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0};
  int hullCount = 0;
  std::vector<double> dst(src.size(), 0.0);
  auto data = buildTaskData(src, dst, hullCount);
  shvedova_v_graham_convex_hull_seq::GrahamConvexHullSequential task(data);
  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  EXPECT_FALSE(task.RunImpl());
}

TEST(shvedova_v_graham_convex_hull_seq, convex_collinear_horizontal) {
  std::vector<double> src = {0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0};
  int hullCount = 0;
  std::vector<double> dst(src.size(), 0.0);
  auto data = buildTaskData(src, dst, hullCount);
  shvedova_v_graham_convex_hull_seq::GrahamConvexHullSequential task(data);
  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  EXPECT_FALSE(task.RunImpl());
}

TEST(shvedova_v_graham_convex_hull_seq, convex_invalid_too_few) {
  std::vector<double> src = {0.0, 0.0, 1.0, 1.0};
  int hullCount = 0;
  std::vector<double> dst(src.size(), 0.0);
  auto data = buildTaskData(src, dst, hullCount);
  shvedova_v_graham_convex_hull_seq::GrahamConvexHullSequential task(data);
  EXPECT_FALSE(task.ValidationImpl());
}

TEST(shvedova_v_graham_convex_hull_seq, convex_invalid_odd_coords) {
  std::vector<double> src = {0.0, 0.0, 1.0};
  int hullCount = 0;
  std::vector<double> dst(src.size(), 0.0);
  auto data = buildTaskData(src, dst, hullCount);
  shvedova_v_graham_convex_hull_seq::GrahamConvexHullSequential task(data);
  EXPECT_FALSE(task.ValidationImpl());
}

TEST(shvedova_v_graham_convex_hull_seq, convex_rhomb) {
  std::vector<double> src = {2.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, -2.0};
  int hullCount = 0;
  std::vector<double> dst(src.size(), 0.0);
  auto data = buildTaskData(src, dst, hullCount);
  executeTask(data);
  std::vector<double> exp = {0.0, -2.0, 2.0, 0.0, 0.0, 2.0, -2.0, 0.0};
  EXPECT_EQ(static_cast<size_t>(hullCount), exp.size() / 2);
  EXPECT_EQ(std::vector<double>(dst.begin(), dst.begin() + exp.size()), exp);
}

TEST(shvedova_v_graham_convex_hull_seq, convex_square) {
  std::vector<double> src = {2.0, 2.0, -2.0, 2.0, -2.0, -2.0, 2.0, -2.0};
  int hullCount = 0;
  std::vector<double> dst(src.size(), 0.0);
  auto data = buildTaskData(src, dst, hullCount);
  executeTask(data);
  std::vector<double> exp = {2.0, -2.0, 2.0, 2.0, -2.0, 2.0, -2.0, -2.0};
  EXPECT_EQ(static_cast<size_t>(hullCount), exp.size() / 2);
  EXPECT_EQ(std::vector<double>(dst.begin(), dst.begin() + exp.size()), exp);
}

TEST(shvedova_v_graham_convex_hull_seq, convex_rhomb_inner) {
  std::vector<double> src = {0.3, -0.25, 1.0, 0.0, 2.0, 0.0, 0.3, 0.25, 0.0, -2.0,
                             0.0, -1.0, 0.25, -0.3, -0.25, -0.3, 0.0, 1.0, 0.0, 2.0,
                             -0.25, 0.3, 0.25, 0.3, -0.3, 0.25, -1.0, 0.0, -2.0,
                             0.0, -0.3, -0.25, 0.1, 0.1};
  int hullCount = 0;
  std::vector<double> dst(src.size(), 0.0);
  auto data = buildTaskData(src, dst, hullCount);
  executeTask(data);
  std::vector<double> exp = {0.0, -2.0, 2.0, 0.0, 0.0, 2.0, -2.0, 0.0};
  EXPECT_EQ(static_cast<size_t>(hullCount), exp.size() / 2);
  EXPECT_EQ(std::vector<double>(dst.begin(), dst.begin() + exp.size()), exp);
}

TEST(shvedova_v_graham_convex_hull_seq, convex_square_inner_complex) {
  std::vector<double> src = {-2.0, -2.0, -1.0, -1.0, -0.5, -1.0, -1.0, -0.5, 2.0,
                             -2.0, 0.5, -1.0, 1.0, -1.0, 1.0, -0.5, 2.0, 2.0,
                             1.0, 1.0, 0.5, 1.0, 1.0, 0.5, -2.0, 2.0, -0.5,
                             1.0, -1.0, 1.0, -1.0, 0.5, 0.1, 0.1};
  int hullCount = 0;
  std::vector<double> dst(src.size(), 0.0);
  auto data = buildTaskData(src, dst, hullCount);
  executeTask(data);
  std::vector<double> exp = {2.0, -2.0, 2.0, 2.0, -2.0, 2.0, -2.0, -2.0};
  EXPECT_EQ(static_cast<size_t>(hullCount), exp.size() / 2);
  EXPECT_EQ(std::vector<double>(dst.begin(), dst.begin() + exp.size()), exp);
}

TEST(shvedova_v_graham_convex_hull_seq, convex_random) {
  constexpr int kCount = 100;
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_real_distribution<> dist(-100.0, 100.0);
  std::vector<double> src;
  src.reserve(kCount * 2);
  for (int i = 0; i < kCount; ++i) {
    src.push_back(dist(rng));
    src.push_back(dist(rng));
  }
  int hullCount = 0;
  std::vector<double> dst(src.size(), 0.0);
  auto data = buildTaskData(src, dst, hullCount);
  executeTask(data);
  size_t n = static_cast<size_t>(hullCount);
  std::vector<std::pair<double, double>> convHull;
  for (size_t i = 0; i < n; ++i)
    convHull.emplace_back(dst[2 * i], dst[2 * i + 1]);
  for (size_t i = 0; i < n; ++i) {
    double dx = convHull[(i + 1) % n].first - convHull[i].first;
    double dy = convHull[(i + 1) % n].second - convHull[i].second;
    double dx2 = convHull[(i + 2) % n].first - convHull[(i + 1) % n].first;
    double dy2 = convHull[(i + 2) % n].second - convHull[(i + 1) % n].second;
    double cross = dx * dy2 - dy * dx2;
    EXPECT_GT(cross, 0.0);
  }
}
