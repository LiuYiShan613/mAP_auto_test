#include "main.hpp"
#include <omp.h>
#include <stdexcept>

//#define STRINGIFY(x) #x
//#define MACRO_STRINGIFY(x) STRINGIFY(x)

#define DEBUG false
using namespace std;
double get_IOU(std::vector<int> bbox1, std::vector<int> bbox2)
{
    //Computes IOU between two bboxes in the form [left,top,right,bottom]
    double left_overlap   = std::max(bbox1[0], bbox2[0]);
    double top_overlap    = std::max(bbox1[1], bbox2[1]);
    double right_overlap  = std::min(bbox1[2], bbox2[2]);
    double bottom_overlap = std::min(bbox1[3], bbox2[3]);

    double width_overlap  = right_overlap-left_overlap;
    double height_overlap = bottom_overlap-top_overlap;
    if((width_overlap < 0) | (height_overlap < 0 )) return 0;

    double area_overlap = width_overlap*height_overlap;
    double area1 = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1]);
    double area2 = (bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1]);
    if ((area1+area2-area_overlap) != 0) return area_overlap / (area1+area2-area_overlap);
    else return 0.0;
}

vector<vector<int>> bbox_pyarray_to_vec(py::array_t<int> &array)
{
    vector<vector<int>> vec(int(array.shape()[0]), vector<int>(4));
    for(unsigned i = 0; i < array.shape(0); i++)
    {
        for(unsigned j = 0; j < array.shape(1); j++)
        {
            vec[i][j] = *array.data(i, j);
        }
    }
    return vec;
}

std::tuple<std::vector<std::vector<int>>, std::vector<int>, std::vector<int>> associate_map(
    py::array_t<int> &detections,
    py::array_t<int> &trackers,
    float IOU_threshold = 0.2)
{
    vector<vector<int>> det = bbox_pyarray_to_vec(detections);
    vector<vector<int>> trk = bbox_pyarray_to_vec(trackers);
    
    return associate_detections_to_trackers(det, trk, IOU_threshold);
}

std::tuple<std::vector<std::vector<int>>, std::vector<int>, std::vector<int>> associate_detections_to_trackers(
    std::vector<std::vector<int>> & detections,
    std::vector<std::vector<int>> & trackers,
    float IOU_threshold = 0.2)
{
    /*
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    matches: 2
    unmatched_detections: 1
    unmatched_trackers  : 1
    */
    HungarianAlgorithm HungAlgo;
    //IOU_threshold = 0.2;

    std::vector<int> unmatched_detections={}, unmatched_trackers={};
    if(trackers.size() == 0)
    {
        for(size_t dr=0; dr < detections.size(); dr++) unmatched_detections.push_back(dr); 
        return {
            {{-1, -1}},
            unmatched_detections,
            unmatched_trackers
        };
    }
    std::vector<std::vector<double>> IOU_matrix = {};
    for(size_t det=0; det < detections.size(); det++)
    {
        std::vector<double> tmp = {};
        for(size_t trk=0; trk < trackers.size(); trk++){
            tmp.push_back(
                get_IOU(
                    {detections[det][0], detections[det][1], detections[det][2], detections[det][3]},
                    {trackers[trk][0], trackers[trk][1], trackers[trk][2], trackers[trk][3]}
                )*(-1)
            );
        }
        IOU_matrix.push_back(tmp);
    }

    std::vector<int> matched_indices = {};
    if(IOU_matrix.size() > 0)
    {
        HungAlgo.Solve(IOU_matrix, matched_indices);
    }

    for(size_t det = 0; det < detections.size(); det++)
    {
        bool det_check = true;
        for (size_t iou = 0; iou < IOU_matrix.size(); iou++)
        {
            if(matched_indices[iou] < 0) continue;
            if( det == iou ) { det_check = false; break;} 
        }
        if(det_check) unmatched_detections.push_back(det);
    }

    for(size_t trk=0; trk < trackers.size(); trk++)
    {
        bool trk_check = true;
        for (size_t iou = 0; iou < IOU_matrix.size(); iou++)
        {
            if (matched_indices[iou] < 0) continue;
            if( (int)trk == matched_indices[iou]) { trk_check = false; break;}
        }
        if(trk_check) unmatched_trackers.push_back(trk);
    }

    //filter out matched with low IOU
    std::vector<std::vector<int>> matches = {};
    for (size_t iou = 0; iou < IOU_matrix.size(); iou++)
    {
        if (matched_indices[iou] < 0) continue;
        if(IOU_matrix[iou][matched_indices[iou]]*-1 < IOU_threshold)
        {
            unmatched_detections.push_back(iou);
            unmatched_trackers.push_back(matched_indices[iou]);
        }
        else
        {
            matches.push_back({(int)iou, matched_indices[iou]});
        }
    }

    return {
        matches,
        unmatched_detections,
        unmatched_trackers
    };
}

PYBIND11_MODULE(execmap, m) 
{
    m.def("get_IOU"    , & get_IOU    , "");
    m.def("associate_detections_to_trackers", associate_detections_to_trackers, "");
    m.def("associate_map", associate_map, "");
    
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

}
