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


int main(int argc, const char* argv[]) {
	// ファイルの有無を確認
	auto file_path = fs::path{ "c:/data/3dpcp_book_codes/3rdparty/Open3D/examples/test_data/fragment.pcd" };
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
	const std::string output_path = "C:/projects/cct-seiya-kumada/farthest_point_sampling/FarthestPointSampling/sampled.pcd";
	std::cout << sampled_cloud->size() << std::endl;
	pcl::io::savePCDFileBinary(output_path, *sampled_cloud);
	return 1;
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
	// 抽出された点のインデックスを格納する配列
	auto indices = std::vector<int>(k, 0);
	
	const auto cloud_size = std::size(cloud);

	// 抽出した点と各点との距離を格納する行列
	auto distances = cv::Mat(k, cloud_size, CV_32F, 0.0);

	// 乱数を生成し、最初の点を決める。
	const auto index = generate_random_value(1, cloud_size);
	cv::Vec3f farthest_point = cloud[index];

	// 点の登録
	indices[0] = index;

	// 他の点との距離を計算
	auto min_distances = calcualte_distances(farthest_point, cloud); // (1,cloud_size)

	// distancesの0行目に代入する。
	min_distances.row(0).copyTo(distances.row(0));

	for (auto i = 1; i < k; ++i) {
		indices[i] = argmax(min_distances.begin<float>(), min_distances.end<float>());
		farthest_point = cloud[indices[i]];

		const auto tmp = calcualte_distances(farthest_point, cloud);	
		
		// distancesのi行目に代入する。
		tmp.row(0).copyTo(distances.row(i));
	
		// min_distancesの更新
		min_distances = minimum(min_distances, distances.row(i));
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
