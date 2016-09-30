#include <opencv2/core/core.hpp>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/image_label_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <opencv2/imgproc/imgproc.hpp>

namespace{
  const bool NOT_COLOR_IMAGE = false;

  cv::Mat PadImage(cv::Mat &image, int min_size, double value = -1) {
    if (image.rows >= min_size && image.cols >= min_size) {
      return image;
    }
    int top, bottom, left, right;
    top = bottom = left = right = 0;
    if (image.rows < min_size) {
      top = (min_size - image.rows) / 2;
      bottom = min_size - image.rows - top;
    }

    if (image.cols < min_size) {
      left = (min_size - image.cols) / 2;
      right = min_size - image.cols - left;
    }
    cv::Mat big_image;
    if (value < 0) {
      cv::copyMakeBorder(image, big_image, top, bottom, left, right,
                         cv::BORDER_REFLECT_101);
    } else {
      cv::copyMakeBorder(image, big_image, top, bottom, left, right,
                         cv::BORDER_CONSTANT, cv::Scalar(value));
    }
    return big_image;
  }

}

namespace caffe {

template <typename Dtype>
ImageLabelDataLayer<Dtype>::ImageLabelDataLayer(
    const LayerParameter &param) : BasePrefetchingDataLayer<Dtype>(param) {
  std::random_device rand_dev;
  rng_ = new std::mt19937(rand_dev());
}

template <typename Dtype>
ImageLabelDataLayer<Dtype>::~ImageLabelDataLayer() {
  this->StopInternalThread();
  delete rng_;
}

template <typename Dtype>
void ImageLabelDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top) {
  // Load the parameters from the proto
  auto &data_param = this->layer_param_.image_label_data_param();
  image_dir_ = data_param.image_dir();
  label_dir_ = data_param.label_dir();
  batch_size_ = data_param.batch_size();
  label_padding_value_ = data_param.label_background_value();

  switch (data_param.padding()) {
    case ImageLabelDataParameter_Padding_ZERO:
      image_padding_value_ = 0;
      break;
    case ImageLabelDataParameter_Padding_REFLECT:
      image_padding_value_ = -1;
      break;
    default:
      LOG(FATAL) << "Unknown Padding";
  }

  crop_size_ = -1;
  auto transform_param = this->layer_param_.transform_param();
  if (transform_param.has_crop_size()) {
    crop_size_ = transform_param.crop_size();
  }

  // Read the file with filenames and labels
  vector<std::string> image_lines;
  const string& image_list_path = this->layer_param_.image_label_data_param().image_list_path();
  LOG(INFO) << "Opening image list " << image_list_path;
  std::ifstream infile(image_list_path.c_str());
  string filename;
  while (infile >> filename) {
    image_lines.push_back(filename);
  }

  vector<std::string> label_lines;
  const string& label_list_path = this->layer_param_.image_label_data_param().label_list_path();
  LOG(INFO) << "Opening label list " << image_list_path;
  std::ifstream in_label(label_list_path.c_str());
  while (in_label >> filename) {
    label_lines.push_back(filename);
  }

  CHECK_EQ(image_lines.size(), label_lines.size()) << "image_list and label_list must be the same size";

  vector<std::string>::iterator gtIt = label_lines.begin();
  for(vector<std::string>::iterator imIt = image_lines.begin(); imIt != image_lines.end(); ++imIt, ++gtIt) {
    lines_.push_back(std::make_pair(*imIt, *gtIt));
  }

  if (this->layer_param_.image_label_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_label_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() % this->layer_param_.image_label_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(image_dir_ + lines_[lines_id_].first);
  cv_img = PadImage(cv_img, crop_size_, image_padding_value_);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;

  // Read a label, and use it to initialize the label top blob.
  cv::Mat cv_label = ReadImageToCVMat(label_dir_ + lines_[lines_id_].second, NOT_COLOR_IMAGE);
  cv_label = PadImage(cv_label, crop_size_, label_padding_value_);
  CHECK(cv_label.data) << "Could not load " << lines_[lines_id_].second;

  // Use data_transformer to infer the expected blob shape from a cv_image.
  image_shape_ = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(image_shape_);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_label_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  image_shape_[0] = batch_size;
  top[0]->Reshape(image_shape_);

  // label
  label_shape_ = this->data_transformer_->InferBlobShape(cv_label);
  this->transformed_label_.Reshape(label_shape_);
  label_shape_[0] = batch_size_;
  top[1]->Reshape(label_shape_);

  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(image_shape_);
    this->prefetch_[i].label_.Reshape(label_shape_);
  }

  LOG(INFO) << "output data size: " << top[0]->num() << ","
  << top[0]->channels() << "," << top[0]->height() << ","
  << top[0]->width();

  LOG(INFO) << "output label size: " << top[1]->num() << ","
  << top[1]->channels() << "," << top[1]->height() << ","
  << top[1]->width();
}

template <typename Dtype>
void ImageLabelDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
void ImageLabelDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
//  auto& data_param = this->layer_param_.image_label_data_param();
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Pad the image and label according to the crop size
  cv::Mat cv_img = ReadImageToCVMat(image_dir_ + lines_[lines_id_].first, true);
  cv_img = PadImage(cv_img, crop_size_);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;

  cv::Mat cv_label = ReadImageToCVMat(label_dir_ + lines_[lines_id_].second, NOT_COLOR_IMAGE);
  cv_label = PadImage(cv_label, crop_size_);
  CHECK(cv_label.data) << "Could not load " << lines_[lines_id_].second;

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  auto lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size_; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(image_dir_ + lines_[lines_id_].first);
    cv::Mat cv_label = ReadImageToCVMat(label_dir_ + lines_[lines_id_].second, NOT_COLOR_IMAGE);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    CHECK(cv_label.data) << "Could not load " << lines_[lines_id_].second;

    cv_img = PadImage(cv_img, crop_size_, image_padding_value_);
    cv_label = PadImage(cv_label, crop_size_, label_padding_value_);

    read_time += timer.MicroSeconds();
    timer.Start();

    // Apply transformations (mirror, crop...) to the image
    int image_offset = batch->data_.offset(item_id);
    int label_offset = batch->label_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + image_offset);
    this->transformed_label_.set_cpu_data(prefetch_label + label_offset);
    this->data_transformer_->Transform(cv_img, cv_label,
                                       &(this->transformed_data_),
                                       &(this->transformed_label_));
    trans_time += timer.MicroSeconds();

    // prefetch_label[item_id] = lines_[lines_id_].second;
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_label_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }

  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageLabelDataLayer);
REGISTER_LAYER_CLASS(ImageLabelData);

}  // namespace caffe
