#include <cstdio>
#include <cmath>
#include "emd.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include <string>
#include <boost/filesystem.hpp>
#include <map>
namespace fs = boost::filesystem;


float dist(feature_t* F1, feature_t* F2)
{
	auto dX = F1->X - F2->X;
	auto dY = F1->Y - F2->Y;
	auto dZ = F1->Z - F2->Z;
	return sqrt(dX * dX + dY * dY + dZ * dZ);
}

//void sample_1();
//void sample_2();
void sample();

int main(int argc, char* args[])
{
    sample();
	return 0;
}

//void sample_1()
//{
//    /* 分布Pの特徴ベクトル */
//    feature_t f1[4] = { {100,40,22}, {211,20,2}, {32,190,150}, {2,100,100} };
//    /* 分布Qの特徴ベクトル */
//    feature_t f2[3] = { {0,0,0}, {50,100,80}, {255,255,255} };
//    /* 分布Pの重み */
//    float w1[4] = { 0.4, 0.3, 0.2, 0.1  };
//    /* 分布Qの重み */
//    float w2[3] = { 0.5, 0.3, 0.2 };
//    /* 分布Pのシグネチャ */
//    signature_t s1 = { 4, f1, w1 };
//    /* 分布Qのシグネチャ */
//    signature_t s2 = { 3, f2, w2 };
//
//    /* EMDを計算 */
//    float e;
//    e = emd(&s1, &s2, dist, 0, 0);
//    printf("emd = %f\n", e);
//}
//void sample_2()
//{
//    std::vector<feature_t> f1 = { {100, 40, 22}, {211, 20, 2}, {32, 190, 150}, {2, 100, 100}, };
//    std::vector<feature_t> f2 = { {0, 0, 0}, {50, 100, 80}, {255, 255, 255} };
//    std::vector<float> w1 = { 0.4, 0.3, 0.2, 0.1 };
//    std::vector<float> w2 = { 0.5, 0.3, 0.2 };
//    signature_t s1 = { f1.size(), f1.data(), w1.data() };
//    signature_t s2 = { f2.size(), f2.data(), w2.data() };
//
//    float e;
//    e = emd(&s1, &s2, dist, 0, 0);
//    printf("HOGE emd = %f\n", e);
//
//}
//
namespace std
{
    inline bool operator<(const cv::Vec3b& lhs, const cv::Vec3b& rhs)
    {
        if (lhs[0] < rhs[0])
        {
            return true;
        }
        else
        {
            if (lhs[1] < rhs[1])
            {
                return true;
            }
            else
            {
                if (lhs[2] < rhs[2])
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
        }
    }
}

void make_histogram(const cv::Mat& image, std::map<cv::Vec3b, int>& histo)
{
    auto ite = image.begin<cv::Vec3b>();
    auto end = image.end<cv::Vec3b>();
    while (ite != end)
    {
        ++histo[*ite];
        ++ite;
    }
}

void check_histogram(const cv::Mat& image, const std::map<cv::Vec3b, int>& histo)
{
    auto size = image.size();
    auto h = size.height;
    auto w = size.width;
    auto b = std::begin(histo);
    auto e = std::end(histo);
    auto d = 0;
    while (b != e)
    {
        d += b->second;
        ++b;
    }
    std::cout << d << std::endl;
    std::cout << w * h << std::endl;

}

void make_inputs(const std::map<cv::Vec3b, int>& histo, const float& total_pixels, std::vector<feature_t>& features, std::vector<float>& weights)
{
    auto b = std::begin(histo);
    auto e = std::end(histo);
    while (b != e)
    {
        const auto& rgb = b->first;
        features.emplace_back(feature_t{ static_cast<float>(rgb[0]), static_cast<float>(rgb[1]), static_cast<float>(rgb[2]) });
        weights.emplace_back(b->second / total_pixels);
        ++b;
    }
}

void make_inputs_for_emd(const std::string& path, std::vector<feature_t>& features, std::vector<float>& weights)
{
    auto image = cv::imread(path);
    auto size = image.size();
    auto h = size.height;
    auto w = size.width;

    std::map<cv::Vec3b, int> histo{};
    make_histogram(image, histo);
    //check_histogram(image_0, histo_0);
    float total_pixels = w * h;
    make_inputs(histo, total_pixels, features, weights);
    auto p = fs::path(path);
    std::cout << "> " << fs::basename(p) << std::endl;
    std::cout << " * total_pixels: " << total_pixels << std::endl;
    std::cout << " * features.size(): " << features.size() << std::endl;
}

void sample()
{
    std::string path_1 = "C:\\projects\\cct-seiya-kumada\\earth_movers_distance\\images\\sea_1.png";
    std::vector<feature_t> features_1{};
    std::vector<float> weights_1{};
    make_inputs_for_emd(path_1, features_1, weights_1);
 
    std::string path_2 = "C:\\projects\\cct-seiya-kumada\\earth_movers_distance\\images\\sea_2.png";
    std::vector<feature_t> features_2{};
    std::vector<float> weights_2{};
    make_inputs_for_emd(path_2, features_2, weights_2);

    std::string path_3 = "C:\\projects\\cct-seiya-kumada\\earth_movers_distance\\images\\mountain_1.png";
    std::vector<feature_t> features_3{};
    std::vector<float> weights_3{};
    make_inputs_for_emd(path_3, features_3, weights_3);

    std::string path_4 = "C:\\projects\\cct-seiya-kumada\\earth_movers_distance\\images\\mountain_2.png";
    std::vector<feature_t> features_4{};
    std::vector<float> weights_4{};
    make_inputs_for_emd(path_4, features_4, weights_4);


    signature_t s1 = { features_1.size(), features_1.data(), weights_1.data() };
    signature_t s2 = { features_2.size(), features_2.data(), weights_2.data() };
    auto e12 = emd(&s1, &s2, dist, 0, 0);
    std::cout << " * emd 1-2: " << e12 << std::endl;
   
    signature_t s3 = { features_3.size(), features_3.data(), weights_3.data() };
    auto e13 = emd(&s1, &s3, dist, 0, 0);
    std::cout << " * emd 1-3: " << e13 << std::endl;

    auto e23 = emd(&s2, &s3, dist, 0, 0);
    std::cout << " * emd 2-3: " << e23 << std::endl;

    signature_t s4 = { features_4.size(), features_4.data(), weights_4.data() };
    auto e34 = emd(&s3, &s4, dist, 0, 0);
    std::cout << " * emd 3-4: " << e34 << std::endl;
}
