#include <pcl/io/pcd_io.h>
#include <memory>
#include <filesystem>
#include <vector>
#include <iostream>
#include <random>
#include <opencv2/opencv.hpp>
#include <boost/range/adaptor/indexed.hpp>
#include <vector>
namespace fs = std::filesystem;

auto execute_farthest_point_sampling(const std::vector<cv::Vec3f>& cloud, int k) -> std::vector<cv::Vec3f>;
auto convert_to_vec(const pcl::PointCloud<pcl::PointXYZ>::Ptr pc) -> std::vector<cv::Vec3f>;
auto convert_to_pcd(const std::vector<cv::Vec3f>& ps) -> pcl::PointCloud<pcl::PointXYZ>::Ptr;
auto select_points_with_random(const std::vector<cv::Vec3f>& cloud, int k) -> std::vector<cv::Vec3f>;

const std::string INPUT_PATH = "c:/data/3dpcp_book_codes/3rdparty/Open3D/examples/test_data/fragment.pcd";
const std::string OUTPUT_PATH_1 = "C:/projects/cct-seiya-kumada/farthest_point_sampling/FarthestPointSampling/sampled.pcd";
const std::string OUTPUT_PATH_2 = "C:/projects/cct-seiya-kumada/farthest_point_sampling/FarthestPointSampling/random_sampled.pcd";

int main(int argc, const char* argv[]) {
	// ファイルの有無を確認
	auto file_path = fs::path{ INPUT_PATH };
	if (!fs::exists(file_path)) {
		std::cout << "> The file does not exist!\n";
		return 0;
	}

	// ファイルから点群を読み込む。
	auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
	pcl::io::loadPCDFile(file_path.string(), *cloud);
	if (!cloud->empty()) {
		std::cout << "> Loading is successful!" << std::endl;
		std::cout << "> Cloud size is " << cloud->size() << std::endl;
	}
	else {
		std::cout << "> Loading is failed" << std::endl;
		return 0;
	}

	// std::vector<cv::Vec3f>に変換
	const auto vs = convert_to_vec(cloud);
	
	// FPSを実行
	const auto k = 1000;
	auto results = execute_farthest_point_sampling(vs, k);
	auto sampled_cloud = convert_to_pcd(results);
	std::cout << "> Sampled points size is " << sampled_cloud->size() << std::endl;

	// 保存する。
	std::cout << sampled_cloud->size() << std::endl;
	pcl::io::savePCDFileBinary(OUTPUT_PATH_1, *sampled_cloud);

	// 乱数で点を選択する。
	auto random_results = select_points_with_random(vs, k);
	auto random_cloud = convert_to_pcd(random_results);
	pcl::io::savePCDFileBinary(OUTPUT_PATH_2, *random_cloud);

	return 1;
}

auto select_points_with_random(const std::vector<cv::Vec3f>& cloud, int k) -> std::vector<cv::Vec3f> {
	auto rnd = std::random_device{};
	auto mt = std::mt19937{ rnd()};
	auto dist = std::uniform_int_distribution<>{ 0, static_cast<int>(std::size(cloud)) };
	mt.seed(1);

	auto results = std::vector<cv::Vec3f>{};
	results.reserve(k);
	for (auto i = 0; i < k; ++i) {
		auto index = dist(mt);
		results.emplace_back(cloud[index]);
	}
	return results;
}

auto generate_random_value(int seed, int size) -> int {
	auto rnd = std::random_device{};
	auto mt = std::mt19937{ rnd()};
	auto dist = std::uniform_int_distribution<>{ 0, size };
	mt.seed(1);
	auto index = dist(mt);
	return index;
}

auto calcualte_distances(const cv::Vec3f& p, const std::vector<cv::Vec3f>& cloud) -> cv::Mat;
auto argmax(const std::vector<float>& vs) -> int;
auto argmax(cv::MatIterator_<float> beg, cv::MatIterator_<float> end) -> int;
auto minimum(const cv::Mat& a, const cv::Mat& b) -> cv::Mat;

auto execute_farthest_point_sampling(const std::vector<cv::Vec3f>& cloud, int k) -> std::vector<cv::Vec3f> {
	// 選択された点のインデックスを格納する配列（k個の0で初期化）
	auto indices = std::vector<int>(k, 0);

	// 入力点群の点数
	const auto cloud_size = std::size(cloud);

	// 乱数を生成し、最初の点を決める。
	const auto index = generate_random_value(1, cloud_size);
	cv::Vec3f farthest_point = cloud[index];

	// 点の登録
	indices[0] = index;

	// 他の点との距離を計算
	auto min_distances = calcualte_distances(farthest_point, cloud); // (1,cloud_size)

	for (auto i = 1; i < k; ++i) {
		// 最も距離の長い点のインデックスを見つけ、indicesに登録する。
		indices[i] = argmax(min_distances.begin<float>(), min_distances.end<float>());

		// 新たな最遠点
		farthest_point = cloud[indices[i]];

		// 他の点との距離を計算
		const auto tmp = calcualte_distances(farthest_point, cloud);	
		
		// min_distancesの更新
		min_distances = minimum(min_distances, tmp);
	}

	std::vector<cv::Vec3f> new_cloud{};
	new_cloud.reserve(k);
	for (auto& index : indices) {
		new_cloud.emplace_back(cloud[index]);
	}
	return new_cloud;
}

auto convert_to_vec(const pcl::PointCloud<pcl::PointXYZ>::Ptr pc) -> std::vector<cv::Vec3f> {
	const auto size = pc->size();
	auto vs = std::vector<cv::Vec3f>{};
	vs.reserve(size);
	for (auto i = 0; i < size; ++i) {
		const auto& p = (*pc)[i];
		vs.emplace_back(p.x, p.y, p.z);
	}
	return vs;
}

auto convert_to_pcd(const std::vector<cv::Vec3f>& ps) -> pcl::PointCloud<pcl::PointXYZ>::Ptr {
	auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
	cloud->reserve(std::size(ps));
	for (const auto& p : ps) {
		cloud->emplace_back(p[0], p[1], p[2]);
	}
	return cloud;
}

auto calcualte_distances(const cv::Vec3f& p, const std::vector<cv::Vec3f>& cloud) -> cv::Mat {
	auto ms = cv::Mat(1, std::size(cloud), CV_32F, 0.0);
	for (auto i = 0; i < std::size(cloud); ++i) {
		const auto a = p - cloud[i];
		ms.at<float>(0, i) = cv::norm(a);
	}
	return ms;
}

auto argmax(cv::MatIterator_<float> beg, cv::MatIterator_<float> end) -> int {
	auto ite = std::max_element(beg, end);
	auto argmax_val = std::distance(beg, ite);
	return argmax_val;
}

auto argmax(const std::vector<float>& vs) -> int {
	auto ite = std::max_element(std::begin(vs), std::end(vs));
	auto argmax_val = std::distance(std::begin(vs), ite);
	return argmax_val;
}

auto minimum(const cv::Mat& a, const cv::Mat& b) -> cv::Mat {
	auto c = a.clone();
	for (auto i = 0; i < a.cols; ++i) {
		c.at<float>(0,i) = std::min(a.at<float>(0, i), b.at<float>(0, i));
	}
	return c;
}
