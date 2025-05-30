#include "tbb/deryabin_m_hoare_sort_simple_merge/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

#include "oneapi/tbb/parallel_for.h"

void deryabin_m_hoare_sort_simple_merge_tbb::HoaraSort(std::vector<double>& a, size_t first, size_t last) {
  if (first >= last) {
    return;
  }
  size_t i = first;
  size_t j = last;
  double tmp = 0;
  double x =
      std::max(std::min(a[first], a[(first + last) / 2]),
               std::min(std::max(a[first], a[(first + last) / 2]),
                        a[last]));  // выбор опорного элемента как медианы первого, среднего и последнего элементов
  do {
    while (a[i] < x) {
      i++;
    }
    while (a[j] > x) {
      j--;
    }
    if (i < j && a[i] > a[j]) {
      tmp = a[i];
      a[i] = a[j];
      a[j] = tmp;
    }
  } while (i < j);
  HoaraSort(a, i + 1, last);
  HoaraSort(a, first, j);
}

void deryabin_m_hoare_sort_simple_merge_tbb::MergeTwoParts(std::vector<double>& a, size_t left, size_t right,
                                                           size_t dimension) {
  size_t middle = (right - left) / 2;
  size_t l_cur = 0;
  size_t r_cur = 0;
  std::vector<double> l_buff(middle + 1);
  std::vector<double> r_buff(middle + 1);
  std::copy(a.begin() + (long)left, a.begin() + (long)left + (long)middle + 1, l_buff.begin());
  std::copy(a.begin() + (long)left + (long)middle + 1, a.begin() + (long)right + 1, r_buff.begin());
  for (size_t i = left; i <= right; i++) {
    if (l_cur <= middle && r_cur <= middle) {
      if (l_buff[l_cur] < r_buff[r_cur]) {
        a[i] = l_buff[l_cur];
        l_cur++;
      } else {
        a[i] = r_buff[r_cur];
        r_cur++;
      }
    } else if (l_cur <= middle) {
      a[i] = l_buff[l_cur];
      l_cur++;
    } else {
      a[i] = r_buff[r_cur];
      r_cur++;
    }
  }
}

bool deryabin_m_hoare_sort_simple_merge_tbb::HoareSortTaskSequential::PreProcessingImpl() {
  input_array_A_ = reinterpret_cast<std::vector<double>*>(task_data->inputs[0])[0];
  dimension_ = task_data->inputs_count[0];
  chunk_count_ = task_data->inputs_count[1];
  min_chunk_size_ = dimension_ / chunk_count_;
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_tbb::HoareSortTaskSequential::ValidationImpl() {
  return static_cast<unsigned short>(task_data->inputs_count[0]) > 2 &&
         static_cast<unsigned short>(task_data->inputs_count[1]) >= 2 &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool deryabin_m_hoare_sort_simple_merge_tbb::HoareSortTaskSequential::RunImpl() {
  size_t count = 0;
  size_t chunk_count = chunk_count_;
  while (count != chunk_count_) {
    HoaraSort(input_array_A_, count * min_chunk_size_, ((count + 1) * min_chunk_size_) - 1);
    count++;
  }
  for (size_t i = 0; i < (size_t)(log((double)chunk_count_) / std::numbers::ln2); i++) {
    for (size_t j = 0; j < chunk_count; j++) {
      MergeTwoParts(input_array_A_, j * min_chunk_size_ << (i + 1), ((j + 1) * min_chunk_size_ << (i + 1)) - 1,
                    dimension_);
      chunk_count--;
    }
  }
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_tbb::HoareSortTaskSequential::PostProcessingImpl() {
  reinterpret_cast<std::vector<double>*>(task_data->outputs[0])[0] = input_array_A_;
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_tbb::HoareSortTaskTBB::PreProcessingImpl() {
  input_array_A_ = reinterpret_cast<std::vector<double>*>(task_data->inputs[0])[0];
  dimension_ = task_data->inputs_count[0];
  chunk_count_ = task_data->inputs_count[1];
  min_chunk_size_ = dimension_ / chunk_count_;
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_tbb::HoareSortTaskTBB::ValidationImpl() {
  return static_cast<unsigned short>(task_data->inputs_count[0]) > 2 &&
         static_cast<unsigned short>(task_data->inputs_count[1]) >= 2 &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool deryabin_m_hoare_sort_simple_merge_tbb::HoareSortTaskTBB::RunImpl() {
  oneapi::tbb::parallel_for(0, (int)chunk_count_, 1, [=, this](int count) {
    HoaraSort(input_array_A_, count * min_chunk_size_, ((count + 1) * min_chunk_size_) - 1);
  });
  for (size_t i = 0; i < (size_t)(log((double)chunk_count_) / std::numbers::ln2); i++) {
    oneapi::tbb::parallel_for(0, (int)chunk_count_ >> (i + 1), 1, [=, this](int j) {
      MergeTwoParts(input_array_A_, (size_t)j * min_chunk_size_ << (i + 1),
                    (((size_t)j + 1) * min_chunk_size_ << (i + 1)) - 1, dimension_);
    });
  }
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_tbb::HoareSortTaskTBB::PostProcessingImpl() {
  reinterpret_cast<std::vector<double>*>(task_data->outputs[0])[0] = input_array_A_;
  return true;
}
