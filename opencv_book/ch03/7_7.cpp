#include "7_7.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include "util.h"
#include <filesystem>
namespace fs = std::filesystem;

enum class Joints {
    NOSE,
    SPINE_SHOULDER,
    SHOULDER_RIGHT,
    ELBOW_RIGHT,
    HAND_RIGHT,
    SHOULDER_LEFT,
    ELBOW_LEFT,
    HAND_LEFT,
    HIP_RIGHT,
    KNEE_RIGHT,
    FOOT_RIGHT,
    HIP_LEFT,
    KNEE_LEFT,
    FOOT_LEFT,
    EYE_RIGHT,
    EYE_LEFT,
    EAR_RIGHT,
    EAR_LEFT,
};

const auto BONES = std::vector<std::pair<Joints, Joints>>{
    std::make_pair(Joints::SPINE_SHOULDER, Joints::SHOULDER_RIGHT),
    std::make_pair(Joints::SPINE_SHOULDER, Joints::SHOULDER_LEFT),
    std::make_pair(Joints::SHOULDER_RIGHT, Joints::ELBOW_RIGHT),
    std::make_pair(Joints::ELBOW_RIGHT,    Joints::HAND_RIGHT),
    std::make_pair(Joints::SHOULDER_LEFT,  Joints::ELBOW_LEFT),
    std::make_pair(Joints::ELBOW_LEFT,     Joints::HAND_LEFT),
    std::make_pair(Joints::SPINE_SHOULDER, Joints::HIP_RIGHT),
    std::make_pair(Joints::HIP_RIGHT,      Joints::KNEE_RIGHT),
    std::make_pair(Joints::KNEE_RIGHT,     Joints::FOOT_RIGHT),
    std::make_pair(Joints::SPINE_SHOULDER, Joints::HIP_LEFT),
    std::make_pair(Joints::HIP_LEFT,       Joints::KNEE_LEFT),
    std::make_pair(Joints::KNEE_LEFT,      Joints::FOOT_LEFT),
    std::make_pair(Joints::SPINE_SHOULDER, Joints::NOSE),
    std::make_pair(Joints::NOSE,           Joints::EYE_RIGHT),
    std::make_pair(Joints::EYE_RIGHT,      Joints::EAR_RIGHT),
    std::make_pair(Joints::NOSE,           Joints::EYE_LEFT),
    std::make_pair(Joints::EYE_LEFT,       Joints::EAR_LEFT)
};

auto get_colors() -> std::vector<cv::Scalar> {
    return std::vector<cv::Scalar>{
        cv::Scalar{ 255, 0, 0 },   cv::Scalar{ 255, 85, 0 },  cv::Scalar{ 255, 170, 0 }, cv::Scalar{ 255, 255, 0 }, cv::Scalar{ 170, 255, 0 },
        cv::Scalar{ 85, 255, 0 },  cv::Scalar{ 0, 255, 0 },   cv::Scalar{ 0, 255, 85 },  cv::Scalar{ 0, 255, 170 }, cv::Scalar{ 0, 255, 255 },
        cv::Scalar{ 0, 170, 255 }, cv::Scalar{ 0, 85, 255 },  cv::Scalar{ 0, 0, 255 },   cv::Scalar{ 85, 0, 255 },  cv::Scalar{ 170, 0, 255 },
        cv::Scalar{ 255, 0, 255 }, cv::Scalar{ 255, 0, 170 }, cv::Scalar{ 255, 0, 85 }
    };
}

void lightweight_openpose() {
	// ファイルを読み込む。
	const auto IMAGE_PATH = std::string{"C:\\projects\\opencv_book\\opencv_dl_book\\ch7\\7.7\\lightweight-openpose\\pose.jpg"};
	const auto is_good_image = load_image(IMAGE_PATH);
	if (!is_good_image) {
		std::cout << "invalid file path\n";
		return;
	}
	const cv::Mat image = is_good_image.value();

	// 重みファイルパスを確認する。
	const auto WEIGHTS_PATH = std::string{ "C:\\data\\opencv_book\\7.7\\lightweight-openpose\\human-pose-estimation.onnx" };
	if (!fs::exists(WEIGHTS_PATH)) {
		std::cout << "invalid model path\n";
		return;
	}
    auto model = cv::dnn::KeypointsModel(WEIGHTS_PATH);

}
