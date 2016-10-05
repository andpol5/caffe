#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_weighted_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace{
  const float MIN_GRADIENT = 1e-10;
  // TODO: generalize this into a parameter
  const std::vector<int> CLASS_LABELS = {0, 1};
}

namespace caffe {

template <typename Dtype>
void SoftmaxWithWeightedLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
  const vector<int>& labelShape = bottom[1]->shape();
  pixel_weights_.Reshape(labelShape);
}

template <typename Dtype>
void  SoftmaxWithWeightedLossLayer<Dtype>::recalculate_pixel_weights(const Blob<Dtype>& label){
  // Zero out the weights blob
  caffe_set<Dtype>(pixel_weights_.count(), static_cast<Dtype>(0), pixel_weights_.mutable_cpu_data());

  // Since there are only 2 labels: 0-background and 1-data, we can just sum (count) the data
  Dtype label_count = label.asum_data();
  Dtype background_count = label.count() - label_count;

  Dtype label_weight = static_cast<Dtype>(1.0 / label_count);
  Dtype background_weight = static_cast<Dtype>(1.0 / background_count);

  const Dtype* label_ptr = label.cpu_data();
  Dtype* weight_ptr = pixel_weights_.mutable_cpu_data();
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label_ptr[i * inner_num_ + j]);
      if(label_value == 0){
        weight_ptr[i * inner_num_ + j] = background_weight;
      }
      else{
        weight_ptr[i * inner_num_ + j] = label_weight;
      }
    }
  }
}

template <typename Dtype>
void SoftmaxWithWeightedLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
  pixel_weights_.Reshape(bottom[1]->shape());
}

template <typename Dtype>
Dtype SoftmaxWithWeightedLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void SoftmaxWithWeightedLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  recalculate_pixel_weights(*bottom[1]);
  const Dtype* weight = pixel_weights_.cpu_data();
  const int dim = prob_.count() / outer_num_;
  Dtype loss = static_cast<Dtype>(0);
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      loss -= weight[i * inner_num_ + j] * log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                           static_cast<Dtype>(FLT_MIN)));
    }
  }
//  Dtype norm = get_normalizer(normalization_, count);
//  if (norm < MIN_GRADIENT) {
//    loss = 0;
//  } else {
//    loss = loss / norm;
//  }
  top[0]->mutable_cpu_data()[0] = loss;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithWeightedLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    const Dtype* weight = pixel_weights_.cpu_data();
    int dim = prob_.count() / outer_num_;
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        const Dtype weight_value = static_cast<Dtype>(weight[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } else {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            if (c == label_value) {
              bottom_diff[i * dim + c * inner_num_ + j] -=1;
            }
            bottom_diff[i * dim + c * inner_num_ + j] *= weight_value;
          }
        }
      }
    }
//    // Scale gradient
//    Dtype norm = get_normalizer(normalization_, count);
//    if (norm > static_cast<Dtype>(MIN_GRADIENT)) {
//      Dtype loss_weight = top[0]->cpu_diff()[0] / norm;
//      caffe_scal(prob_.count(), loss_weight, bottom_diff);
//    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithWeightedLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithWeightedLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithWeightedLoss);

}  // namespace caffe
