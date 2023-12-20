#include "7_7.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include "util.h"
#include <filesystem>
#include <boost/range/adaptor/indexed.hpp>
#include <cmath>
#include <numbers>
namespace fs = std::filesystem;

enum class Joints:int {
    NOSE = 0,
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

auto radians_to_degrees(double radians) -> double {
    return radians * (180.0 / std::numbers::pi);
}

auto draw_bone(
    const cv::Mat& image, 
    const cv::Point2f& start_point, 
    const cv::Point2f& end_point, 
    const cv::Scalar& color,
    int thickness=4) {

    const auto mean_x = static_cast<int>((start_point.x + end_point.x) / 2.0);
    const auto mean_y = static_cast<int>((start_point.y + end_point.y) / 2.0);
    const auto center = cv::Point{ mean_x, mean_y };
    const auto diff = start_point - end_point;
    const auto length = std::sqrt(diff.x * diff.x + diff.y * diff.y);
    const auto axes = cv::Size{ static_cast<int>(length / 2.0), thickness };
    const auto angle = static_cast<int>(radians_to_degrees(std::atan2(diff.y, diff.x)));
    auto polygon = std::vector<cv::Point>{};
    cv::ellipse2Poly(center, axes, angle, 0, 360, 1, polygon);
    cv::fillConvexPoly(image, polygon, color, cv::LINE_AA);
}

void lightweight_openpose() {
    // ファイルを読み込む。
    const auto IMAGE_PATH = std::string{ "C:\\projects\\opencv_book\\opencv_dl_book\\ch7\\7.7\\lightweight-openpose\\pose_2.jpg" };
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

    // モデルの推論に使用するエンジンとデバイスを設定する
    model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // モデルの入力パラメーターを設定する
    const auto scale = 1.0 / 255.0;                       //スケールファクター
    const auto size = cv::Size{ 256, 456 };               // 入力サイズ（仮）
    const auto mean = cv::Scalar{ 128.0, 128.0, 128.0 };  // 差し引かれる平均値
    const auto swap = false;                              // チャンネルの順番（True: RGB、False: BGR）
    const auto crop = false;                              // クロップ
    model.setInputParams(scale, size, mean, swap, crop);

    // カラーテーブルを取得する。
    const auto colors = get_colors();

    // モデルの入力サイズを設定する
    const auto rows = static_cast<float>( image.rows );
    const auto cols = static_cast<float>( image.cols );
    const auto input_size = cv::Size{ 256, int((256 / cols) * rows) };  // アスペクト比を保持する
    std::cout << input_size << std::endl;
    model.setInputSize(input_size);

    // キーポイントを検出する
    const auto confidence_threshold = 0.6;
    const auto keypoints = model.estimate(image, confidence_threshold);

    // キーポイントを描画する
    for (const auto& p : keypoints | boost::adaptors::indexed(0)) {
        const auto& index = p.index();
        const auto& keypoint = p.value();
        // point = tuple(map(int, keypoint.tolist()))
        const auto& radius = 5;
        const auto& color = colors[index];
        const auto thickness = -1;
        cv::circle(image, keypoint, radius, color, thickness, cv::LINE_AA);
    }

    // ボーンを描画する
    for (const auto& bone : BONES) {
        const int i0 = static_cast<int>(std::get<0>(bone));
        const int i1 = static_cast<int>(std::get<1>(bone));
        const auto& point1 = keypoints[i0];
        const auto& point2 = keypoints[i1];
        if ((point1 == cv::Point2f{ -1, -1 }) || (point2 == cv::Point2f{ -1, -1 })) {
            continue;
        }
        draw_bone(image, point1, point2, colors[i1]);
    }

   	// 描画する。
    cv::imshow("image", image);
	const auto output_path = std::string{ "C:\\data\\opencv_book\\7.7_outputs\\result.jpg" };
	cv::imwrite(output_path, image);
	cv::waitKey(0);
}
