#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class CropLayer : public cv::dnn::Layer {
private:
	mutable int xstart_;
	mutable int xend_;
	mutable int ystart_;
	mutable int yend_;

public:
	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params);
	CropLayer(const cv::dnn::LayerParams& params);
	CropLayer(const CropLayer&) = delete;
	CropLayer& operator=(const CropLayer&) = delete;

	virtual void forward(
		std::vector<cv::Mat*>&	inputs,
		std::vector<cv::Mat>&	outputs,
		std::vector<cv::Mat>&	internals) override;

	virtual bool getMemoryShapes(
		const std::vector<cv::dnn::MatShape>&	inputs,
		const int								requiredOutputs, 
		std::vector<cv::dnn::MatShape>&			outputs,
		std::vector<cv::dnn::MatShape>&			internals) const override;
};
