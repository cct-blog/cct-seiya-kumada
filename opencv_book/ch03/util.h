#pragma once
#include <string>
#include <optional>
#include <opencv2/opencv.hpp>

auto load_image(const std::string& path) -> std::optional<cv::Mat>;
auto read_coco_file(const std::string& path) -> std::optional<std::vector<std::string>>;
auto make_color_table(int num) -> std::vector<cv::Scalar>;
