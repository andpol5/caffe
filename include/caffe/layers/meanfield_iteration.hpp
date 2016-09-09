#ifndef CAFFE_MF_ITERATION_HPP_
#define CAFFE_MF_ITERATION_HPP_
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/split_layer.hpp"

#include "caffe/util/modified_permutohedral.hpp"

namespace caffe {

    // Forward declare MultiStageMeanfieldLayer
    template <typename Dtype>
    class MultiStageMeanfieldLayer;

    template <typename Dtype>
    class MeanfieldIteration {

    public:

        bool is_first_iteration_; // TODO: a nasty hack, fix later.

        /**
         * Every MeanfieldIteration must belong to a {@link MultiStageMeanfieldLayer}.
         */
        explicit MeanfieldIteration(MultiStageMeanfieldLayer<Dtype> * const msmf_parent) :
                is_first_iteration_(false), msmf_parent_(msmf_parent) { }

        /**
         * Must be invoked only once after the construction of the layer.
         */
        void OneTimeSetUp(
                Blob<Dtype> * const unary_terms,
                Blob<Dtype> * const softmax_input,
                Blob<Dtype> * const output_blob,
                const shared_ptr<ModifiedPermutohedral> & spatial_lattice,
                const Blob<Dtype> * const spatial_norm);

        /**
         * Must be invoked before invoking {@link Forward_cpu()}
         */
        void PrePass(
                const vector<shared_ptr<Blob<Dtype> > > &  parameters_to_copy_from,
                const vector<shared_ptr<ModifiedPermutohedral> > * bilateral_lattices,
                const Blob<Dtype> * const bilateral_norms);

        /**
         * Forward pass - to be called during inference.
         */
        void Forward_cpu();
        void Forward_gpu();

        /**
         * Backward pass - to be called during training.
         */
        void Backward_cpu();
        void Backward_gpu();

        // A quick hack. This should be properly encapsulated.
        vector<shared_ptr<Blob<Dtype> > >& blobs() {
            return blobs_;
        }

    private:

        vector<shared_ptr<Blob<Dtype> > > blobs_;

        MultiStageMeanfieldLayer<Dtype> * const msmf_parent_;
        int count_;
        int num_;
        int channels_;
        int height_;
        int width_;
        int num_pixels_;

        Blob<Dtype> spatial_out_blob_;
        Blob<Dtype> bilateral_out_blob_;
        Blob<Dtype> pairwise_;
        Blob<Dtype> prob_;
        Blob<Dtype> message_passing_;

        vector<Blob<Dtype>*> softmax_top_vec_;
        vector<Blob<Dtype>*> softmax_bottom_vec_;
        vector<Blob<Dtype>*> sum_top_vec_;
        vector<Blob<Dtype>*> sum_bottom_vec_;

        shared_ptr<SoftmaxLayer<Dtype> > softmax_layer_;
        shared_ptr<EltwiseLayer<Dtype> > sum_layer_;

        shared_ptr<ModifiedPermutohedral> spatial_lattice_;
        const vector<shared_ptr<ModifiedPermutohedral> > * bilateral_lattices_;
        const Blob<Dtype>* spatial_norm_;
        const Blob<Dtype>* bilateral_norms_;

    };

}//namespace caffe
#endif
