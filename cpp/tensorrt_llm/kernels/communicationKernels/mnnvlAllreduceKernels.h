/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TRTLLM_MNNVL_ALLREDUCE_KERNELS_H
#define TRTLLM_MNNVL_ALLREDUCE_KERNELS_H

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/tllmDataType.h"
#include "tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.h"
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::mnnvl
{

/**
 * \brief Parameters for MNNVL (Multi-Node NVLink) AllReduce fusion operations.
 *
 * \note This struct is used by both oneshotAllreduceFusionOp() and twoshotAllreduceFusionOp()
 *
 * \see oneshotAllreduceFusionOp
 * \see twoshotAllreduceFusionOp
 */
struct AllReduceFusionParams
{
    //! \name Environmental and Auxiliary Data
    //! @{

    int nRanks;                   //!< Total number of participating ranks in the AllReduce operation
    int rank;                     //!< Current rank ID
    tensorrt_llm::DataType dType; //!< Data type of the tensors (e.g., FP16, BF16, FP32)
    int numTokens;                //!< Number of tokens in the input tensor
    int tokenDim;                 //!< Hidden Dimension
    void** bufferPtrsDev;         //!< Unicast Device pointers to communication buffers for each rank
    void* bufferPtrLocal;         //!< Local buffer pointer for temporary storage (i.e., bufferPtrsDev[rank])
    void* multicastPtr;           //!< Multicast buffer pointer.
    uint32_t* bufferFlags;        //!< Synchronization flags for coordinating communication phases
    bool rmsNormFusion;           //!< Whether to fuse RMS normalization with the AllReduce operation
    ar_fusion::AllReduceFusionPattern pattern
        = ar_fusion::AllReduceFusionPattern::kAllReduce; //!< Fused epilogue pattern

    //! @}

    //! \name Input and Output Data
    //! @{

    void const* input;      //!< Input tensor to be reduced across all ranks
    void const* residualIn; //!< Residual input tensor for skip connections (used when rmsnormFusion=true)
    void const* gamma;      //!< Gamma parameters for RMS normalization (used when rmsnormFusion=true)
    double epsilon;         //!< Epsilon value for RMS normalization numerical stability (used when rmsnormFusion=true)

    void* residualOut = nullptr;        //!< Output tensor for residual connection result (used when rmsnormFusion=true)
    void* output = nullptr;             //!< Output tensor containing the AllReduce or RMSNorm result
    void* quantOut = nullptr;           //!< Quantized RMSNorm output (used by quantized fusion patterns)
    void* scaleOut = nullptr;           //!< NVFP4 scale-factor output (used by NVFP4 fusion patterns)
    float const* scaleFactor = nullptr; //!< Quantization scale factor
    QuantizationSFLayout layout = QuantizationSFLayout::SWIZZLED; //!< NVFP4 scale-factor layout
    cudaStream_t stream;                                          //!< CUDA stream for asynchronous kernel execution

    //! @}
};

void oneshotAllreduceFusionOp(AllReduceFusionParams const& params);
void twoshotAllreduceFusionOp(AllReduceFusionParams const& params);

/**
 * \brief Parameters for an MNNVL all-reduce whose sum is the update of a wide residual step.
 *
 * The sum `u` of every rank's `input` is formed as `oneshotAllreduceFusionOp` forms it (fp32, ascending rank
 * order, one rounding) and is never written out. It is written into the residual streams in place,
 * `S_k <- bf16(bf16(r_k * S_k) + w_k * u)`, and the streams are then read once or twice, each read
 * `x = bf16(sum_k f_k * S_k)` RMS-normed with its own weight: `bf16(g * bf16(x * rsqrt(mean(x^2) + eps)))`.
 * Every tensor is bf16; the streams hold `numStreams` rows of `tokenDim` side by side per token.
 */
struct AllReduceWideResidualParams
{
    int nRanks;
    int rank;
    int numTokens;
    int tokenDim;
    int numStreams;
    void** bufferPtrsDev;
    void* bufferPtrLocal;
    void* multicastPtr;
    uint32_t* bufferFlags;

    void const* input;        //!< [numTokens, tokenDim], this rank's partial
    void* streams;            //!< [numTokens, numStreams * tokenDim], rows `streamsRowStride` apart; written in place
    int64_t streamsRowStride; //!< Elements between consecutive tokens' stream rows
    void const* retention;    //!< [numStreams]
    void const* write;        //!< [numStreams]
    void const* readFactors;  //!< [numStreams]
    void const* normWeight;   //!< [tokenDim]
    void const* readFactors2; //!< [numStreams], or nullptr for one read
    void const* normWeight2;  //!< [tokenDim], or nullptr for one read
    float epsilon;
    float invTokenDim;        //!< fp32 1 / tokenDim, the factor the variance sum is scaled by
    void* normedOut;          //!< [numTokens, tokenDim]
    void* normedOut2;         //!< [numTokens, tokenDim], or nullptr for one read
    cudaStream_t stream;
};

//! Every row in one kernel: the one-shot all-reduce with the wide-residual step as its epilogue.
void oneshotAllreduceWideResidualOp(AllReduceWideResidualParams const& params);
//! The two-shot all-reduce, then a kernel that waits for each row's sum and runs the wide-residual step.
void twoshotAllreduceWideResidualOp(AllReduceWideResidualParams const& params);
} // namespace kernels::mnnvl

TRTLLM_NAMESPACE_END
#endif // TRTLLM_MNNVL_ALLREDUCE_KERNELS_H
