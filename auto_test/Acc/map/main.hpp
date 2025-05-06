#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <iostream>
#include <algorithm>

#include "Hungarian.hpp"
//#include <mv_table_extract.h>

namespace py = pybind11;

double get_IOU(std::vector<int> bbox1, std::vector<int> bbox2);
std::vector<std::vector<int>> bbox_pyarray_to_vec(py::array_t<int> &detections);
std::tuple<std::vector<std::vector<int>>, std::vector<int>, std::vector<int>> associate_map(
    py::array_t<int> &detections,
    py::array_t<int> &trackers,
    float IOU_threshold);
std::tuple<std::vector<std::vector<int>>, std::vector<int>, std::vector<int>> associate_detections_to_trackers(
    std::vector<std::vector<int>> & detections,
    std::vector<std::vector<int>> & trackers,
    float IOU_threshold);



