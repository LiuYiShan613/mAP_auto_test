#ifndef DIVA_GT_H
#define DIVA_GT_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp> // For JSON parsing
#include <algorithm>
#include <filesystem>

class DIVA_gt {
public:
    enum DS {
        DS_GMOT,
        DS_MOT,
        DS_Imagenet_VID
    };

    explicit DIVA_gt(const std::string& video_path);

    void clean();

    void GMOT(const std::string& gt_path);

    void MOT(const std::string& gt_path);

    void Imagenet_VID(const std::string& gt_path);

    float calc_acc(const std::vector<std::vector<float>>& res_list, int frame_num, bool is_async);

    void draw_gt(cv::Mat& frame, int frame_num, bool is_async);

private:
    std::string video_name;
    std::string gt_name;
    std::string gt_path;

    std::vector<std::vector<std::vector<float>>> gt_result;
    std::vector<std::vector<std::vector<float>>> gt_only_pt;
    int gt_number = 0;
    float gt_acc = 0.0;

    std::vector<std::vector<float>> load_csv(const std::string& filepath);

    nlohmann::json read_json_file(const std::string& file_path);

    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> get_all_label(const std::vector<std::vector<float>>& label, int start_index);

    std::string get_basename(const std::string& path);

    std::string get_parent_directory(const std::string& path);

    std::pair<std::vector<int>, std::vector<int>> associate_map(const std::vector<std::vector<float>>& gt, const std::vector<std::vector<float>>& res, float threshold);
};

#endif // DIVA_GT_H
