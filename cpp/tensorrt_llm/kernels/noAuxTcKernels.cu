/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include "moeTopKFuncs.cuh"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/noAuxTcKernels.h"
#include "tensorrt_llm/kernels/quantization.cuh"
#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <type_traits>

namespace cg = cooperative_groups;
using namespace tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
static constexpr int WARP_SIZE = 32;
static constexpr int NumDeepseekExperts = 256;
static constexpr int MaxSupportedExpertCount = 1024;
static constexpr int NumTopGroupScores = 2;
static constexpr int DefaultMaxNumTopExperts = 8;
static constexpr int MaxSupportedTopExperts = 32;
static constexpr int DefaultMaxNumTopGroups = 4;
static constexpr int LargeMaxNumTopGroups = 8;

static __device__ inline float sigmoid_accurate(float x)
{
    return 0.5f * tanhf(0.5f * x) + 0.5f;
}

template <typename InputT, typename BiasT, typename OutputT, typename IdxT, int MaxNumExperts, bool UseGroups,
    int MaxNumTopExperts = DefaultMaxNumTopExperts, int MaxNumTopGroups = DefaultMaxNumTopGroups>
__device__ __forceinline__ void deepseek_v3_topk_block(InputT* scores, OutputT* topkValues, IdxT* topkIndices,
    BiasT* routingBias, int64_t const numTokens, int64_t const numGroup, int64_t const topkGroup, int64_t const topk,
    int64_t const numExperts, int64_t const numExpertsPerGroup, double const routedScalingFactor, int64_t tokenIdx)
{
    __shared__ float __attribute((aligned(128))) smemScoreSigmoid[MaxNumExperts];
    __shared__ float __attribute((aligned(128))) smemScoreBias[MaxNumExperts];

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WARP_SIZE>(block);

    int32_t laneIdx = threadIdx.x % WARP_SIZE;
    int32_t warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WARP_SIZE, 0);

    static constexpr float invalidScoreFloat = float{-INFINITY};

    topkValues += tokenIdx * topk;
    topkIndices += tokenIdx * topk;

    if constexpr (UseGroups)
    {
        int constexpr NumWarps = MaxNumExperts / WARP_SIZE;
        __shared__ float __attribute((aligned(128))) smemGroupScores[NumWarps];

        auto threadExpert = warpIdx * numExpertsPerGroup + laneIdx;
        bool expertSelected = laneIdx < numExpertsPerGroup;

        auto scoreIdx = tokenIdx * static_cast<int64_t>(numExperts) + threadExpert;
        auto biasVal = expertSelected ? static_cast<float>(routingBias[threadExpert]) : invalidScoreFloat;
        float score = expertSelected ? static_cast<float>(scores[scoreIdx]) : invalidScoreFloat;
        auto scoreSigmoid = sigmoid_accurate(score);
        if (expertSelected)
        {
            smemScoreSigmoid[threadExpert] = scoreSigmoid;
        }
        auto scoreBias = float{scoreSigmoid + float{biasVal}};
        if (expertSelected)
        {
            smemScoreBias[threadExpert] = scoreBias;
        }

        float topExpGroupScores[NumTopGroupScores];
        [[maybe_unused]] int32_t topExpGroupIdx[NumTopGroupScores];
        reduce_topk::reduceTopK(warp, topExpGroupScores, topExpGroupIdx, scoreBias, threadExpert,
            /* minValue */ invalidScoreFloat);

        if (warp.thread_rank() == 0)
        {
            auto groupScore = topExpGroupScores[0] + topExpGroupScores[1];
            smemGroupScores[warpIdx] = groupScore;
        }

        __syncthreads();

        float topScores[MaxNumTopExperts];
        int32_t topExperts[MaxNumTopExperts];

        if (warpIdx == 0)
        {
            float topGroups[MaxNumTopGroups];
            int32_t topGroupIdx[MaxNumTopGroups];
            float groupScore = laneIdx < numGroup ? smemGroupScores[laneIdx] : invalidScoreFloat;
            reduce_topk::reduceTopK(warp, topGroups, topGroupIdx, groupScore, laneIdx,
                /* minValue */ invalidScoreFloat);

            float expertScoreGroup[MaxNumTopGroups];
            int32_t expertIdxGroup[MaxNumTopGroups];
#pragma unroll
            for (int ii = 0; ii < MaxNumTopGroups; ++ii)
            {
                auto groupIdx = topGroupIdx[ii];
                expertIdxGroup[ii] = groupIdx * numExpertsPerGroup + laneIdx;
                expertScoreGroup[ii]
                    = (ii < topkGroup) && expertSelected ? smemScoreBias[expertIdxGroup[ii]] : invalidScoreFloat;
            }

            reduce_topk::reduceTopK(
                warp, topScores, topExperts, expertScoreGroup, expertIdxGroup, /* minValue */ invalidScoreFloat, topk);

            int32_t expertIdx = laneIdx < topk ? topExperts[laneIdx] : MaxNumExperts - 1;
            float scoreNorm = laneIdx < topk ? smemScoreSigmoid[expertIdx] : 0.F;
            auto redNorm = cg::reduce(warp, scoreNorm, cg::plus<float>{});
            auto finalScore = static_cast<OutputT>(scoreNorm * routedScalingFactor / (redNorm + 1e-20));
            if (laneIdx < topk)
            {
                topkValues[laneIdx] = static_cast<OutputT>(finalScore);
                topkIndices[laneIdx] = expertIdx;
            }
        }
    }
    else
    {
        for (int e = threadIdx.x; e < numExperts; e += blockDim.x)
        {
            auto scoreIdx = tokenIdx * static_cast<int64_t>(numExperts) + e;
            auto biasVal = static_cast<float>(routingBias[e]);
            float score = static_cast<float>(scores[scoreIdx]);
            auto scoreSigmoid = sigmoid_accurate(score);
            smemScoreSigmoid[e] = scoreSigmoid;
            smemScoreBias[e] = scoreSigmoid + biasVal;
        }

        __syncthreads();

        float topScores[MaxNumTopExperts];
        int32_t topExperts[MaxNumTopExperts];

        if (warpIdx == 0)
        {
            constexpr int NumChunks = (MaxNumExperts + WARP_SIZE - 1) / WARP_SIZE;
            float localScores[NumChunks];
            int32_t localIdx[NumChunks];
#pragma unroll
            for (int ii = 0; ii < NumChunks; ++ii)
            {
                auto expertIdx = ii * WARP_SIZE + laneIdx;
                localIdx[ii] = expertIdx;
                localScores[ii] = expertIdx < numExperts ? smemScoreBias[expertIdx] : invalidScoreFloat;
            }
            reduce_topk::reduceTopK(warp, topScores, topExperts, localScores, localIdx,
                /* minValue */ invalidScoreFloat, topk);

            int32_t expertIdx = laneIdx < topk ? topExperts[laneIdx] : MaxNumExperts - 1;
            float scoreNorm = laneIdx < topk ? smemScoreSigmoid[expertIdx] : 0.F;
            auto redNorm = cg::reduce(warp, scoreNorm, cg::plus<float>{});
            auto finalScore = static_cast<OutputT>(scoreNorm * routedScalingFactor / (redNorm + 1e-20));
            if (laneIdx < topk)
            {
                topkValues[laneIdx] = static_cast<OutputT>(finalScore);
                topkIndices[laneIdx] = expertIdx;
            }
        }
    }
}

template <typename InputT, typename BiasT, typename OutputT, typename IdxT, int MaxNumExperts, bool UseGroups,
    int MaxNumTopExperts = DefaultMaxNumTopExperts, int MaxNumTopGroups = DefaultMaxNumTopGroups>
__global__ void deepseek_v3_topk_kernel(InputT* scores, OutputT* topkValues, IdxT* topkIndices, BiasT* routingBias,
    int64_t const numTokens, int64_t const numGroup, int64_t const topkGroup, int64_t const topk,
    int64_t const numExperts, int64_t const numExpertsPerGroup, double const routedScalingFactor)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    if constexpr (UseGroups)
    {
        if (threadIdx.x / WARP_SIZE >= numGroup)
        {
            return;
        }
    }
    deepseek_v3_topk_block<InputT, BiasT, OutputT, IdxT, MaxNumExperts, UseGroups, MaxNumTopExperts, MaxNumTopGroups>(
        scores, topkValues, topkIndices, routingBias, numTokens, numGroup, topkGroup, topk, numExperts,
        numExpertsPerGroup, routedScalingFactor, blockIdx.x);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// K3 decode specialization: CTAs [0, M) route 896 experts to top-16, while
// CTAs [M, 2M) independently quantize one 3584-wide BF16 activation row to
// MXFP8 with linear UE8M0 group-32 scales.
static constexpr int KimiK3NumExperts = 896;
static constexpr int KimiK3TopK = 16;
static constexpr int KimiK3HiddenSize = 3584;
static constexpr int MxFp8SfVecSize = 32;
static constexpr int KimiK3QuantThreads = KimiK3HiddenSize / CVT_ELTS_PER_THREAD;

__global__ __launch_bounds__(KimiK3QuantThreads) void kimi_k3_noaux_tc_mxfp8_quant_kernel(float* scores,
    float* routingBias, __nv_bfloat16* hiddenStates, __nv_bfloat16* topkValues, int32_t* topkIndices,
    int64_t* quantizedHiddenStates, int32_t* hiddenStatesScale, int64_t numTokens, double routedScalingFactor)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    if (blockIdx.x < numTokens)
    {
        deepseek_v3_topk_block<float, float, __nv_bfloat16, int32_t, MaxSupportedExpertCount, false,
            MaxSupportedTopExperts>(scores, topkValues, topkIndices, routingBias, numTokens, 1, 1, KimiK3TopK,
            KimiK3NumExperts, KimiK3NumExperts, routedScalingFactor, blockIdx.x);
    }
    else
    {
        using QuantT = __nv_bfloat16;
        using QuantPackedVec = PackedVec<QuantT>;
        static constexpr int CvtNumThreadsPerSf = MxFp8SfVecSize / CVT_ELTS_PER_THREAD;
        int const rowIdx = blockIdx.x - numTokens;
        int const colIdx = threadIdx.x;
        int const numColThreads = KimiK3HiddenSize / CVT_ELTS_PER_THREAD;

        std::optional<int> optionalNumRows = numTokens;
        auto sfOut = cvt_quant_get_sf_out_offset<uint32_t, CvtNumThreadsPerSf>(std::nullopt, rowIdx, colIdx,
            optionalNumRows, KimiK3HiddenSize / MxFp8SfVecSize, reinterpret_cast<uint32_t*>(hiddenStatesScale),
            QuantizationSFLayout::LINEAR);
        int64_t const offset = static_cast<int64_t>(rowIdx) * numColThreads + colIdx;
        QuantPackedVec inVec = reinterpret_cast<QuantPackedVec const*>(hiddenStates)[offset];
        reinterpret_cast<uint64_t*>(quantizedHiddenStates)[offset]
            = cvt_warp_fp16_to_mxfp8<QuantT, MxFp8SfVecSize>(inVec, sfOut);
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    __threadfence();
    __syncthreads();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void invokeKimiK3NoAuxTcMxFp8Quant(float* scores, float* bias, __nv_bfloat16* hidden_states, __nv_bfloat16* topk_values,
    int32_t* topk_indices, int64_t* quantized_hidden_states, int32_t* hidden_states_scale, int64_t const num_tokens,
    double const routed_scaling_factor, cudaStream_t const stream)
{
    cudaLaunchConfig_t config;
    config.gridDim = 2 * num_tokens;
    config.blockDim = KimiK3QuantThreads;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;

    cudaLaunchKernelEx(&config, kimi_k3_noaux_tc_mxfp8_quant_kernel, scores, bias, hidden_states, topk_values,
        topk_indices, quantized_hidden_states, hidden_states_scale, num_tokens, routed_scaling_factor);
    sync_check_cuda_error(stream);
}

// Single-group routing (n_group <= 1), one warp per token. It returns what deepseek_v3_topk_block<..., UseGroups =
// false> returns, bit for bit: the same tanh-form sigmoid, the same packed (biased score, 65535 - expert) order, so
// a tie goes to the lower expert, the same warp sum of the picked unbiased scores and the same double-precision
// scale-and-divide. What changes is how the top-k is found, without one warp-wide reduction per pick:
//  1. tau = the largest lane maximum (high word of the packed key) that at least `topk` lane maxima reach. At
//     least `topk` experts reach it, so every expert of the top-k does.
//  2. The experts that reach tau are the candidates, packed into shared memory by one ballot per register slot.
//  3. With at most 32 candidates, one per lane, a candidate's rank is the number of candidates ordered above it:
//     rank < topk selects it, and the rank is its output slot. More than 32 candidates (heavy ties) sorts each
//     lane's experts and takes the top-k one pick at a time.
static constexpr int NoAuxTcWarpsPerCta = 4;

static __device__ __forceinline__ uint32_t warpReduceMax(uint32_t value)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    return __reduce_max_sync(0xffffffff, value);
#else
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        value = max(value, __shfl_xor_sync(0xffffffff, value, offset));
    }
    return value;
#endif
}

// static_cast<float>(score * routedScalingFactor / (sum + 1e-20)) with double arithmetic, as the routing kernels
// compute it, but on the FP32 pipe for all but a vanishing fraction of inputs. For sum >= 2^-13 the 1e-20 is below
// half an ulp of double(sum), so the double expression is the real quotient Q = score * scale / sum rounded twice
// to 53 bits. scale = scaleHi + scaleLo to 2^-48; an error-free product, the residual of an approximate quotient
// and Fast2Sum give q1 = RN_f(q0 + d), with e its exact rounding error and q0 + d within 2^-42 of Q (relative).
// When e stays 2^-40 * q1 inside the half spacing around q1 (a quarter ulp, the short side, at a power of two),
// Q and the double result both round to q1. Every other input takes the double expression itself.
static __device__ __forceinline__ float noAuxTcRoutedWeight(
    float score, float sum, float scaleHi, float scaleLo, double routedScalingFactor)
{
    if (score >= 0.F && sum >= 0x1p-13F && sum <= 0x1p64F)
    {
        float const productHi = score * scaleHi;
        float const productLo = fmaf(score, scaleLo, fmaf(score, scaleHi, -productHi));
        float const q0 = __fdividef(productHi, sum);
        float const residual = fmaf(-q0, sum, productHi) + productLo;
        float const d = __fdividef(residual, sum);
        float const q1 = q0 + d;
        float const roundingError = d - (q1 - q0);
        if (q1 == 0.F)
        {
            return q1;
        }
        uint32_t const bits = __float_as_uint(q1);
        float const ulp = __uint_as_float((bits & 0x7f800000u) - (23u << 23));
        float const halfSpacing = (bits & 0x007fffffu) == 0u ? 0.25F * ulp : 0.5F * ulp;
        if (fabsf(roundingError) <= halfSpacing - q1 * 0x1p-40F)
        {
            return q1;
        }
    }
    return static_cast<float>(score * routedScalingFactor / (sum + 1e-20));
}

// reduce_topk::TopKRedType<float>::makeCmpVal's packing: the score's order-preserving bits over 65535 - expert, so
// the larger key is the larger score and, between equal scores, the lower expert.
static __device__ __forceinline__ uint64_t packScoreExpert(float score, int32_t expert)
{
    uint32_t const bits = __float_as_uint(score);
    uint32_t const ordered = bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
    return (static_cast<uint64_t>(ordered) << 32)
        | static_cast<uint64_t>(0xffffu & static_cast<uint32_t>(65535 - expert));
}

static __device__ __forceinline__ int32_t unpackExpert(uint64_t key)
{
    return 65535 - static_cast<int32_t>(key & 0xffffu);
}

// AllSlotsValid: numExperts == ExpertsPerLane * WARP_SIZE, so no slot asks for the expert count and every load is
// unconditional. Measured on sm_103 at 512 experts: testing the runtime count in the load-and-score phase, guarded
// or clamped, takes a token from ~2.1 to ~3.7 us at decode sizes, so the full case is its own instantiation.
template <typename InputT, typename BiasT, typename OutputT, typename IdxT, int ExpertsPerLane, bool AllSlotsValid>
__global__ void __launch_bounds__(NoAuxTcWarpsPerCta* WARP_SIZE) noaux_tc_warp_per_token_kernel(InputT const* scores,
    BiasT const* routingBias, OutputT* topkValues, IdxT* topkIndices, int64_t const numTokens, int32_t const numExperts,
    int32_t const topk, double const routedScalingFactor, float const scaleHi, float const scaleLo)
{
    static_assert(
        ExpertsPerLane > 0 && (ExpertsPerLane & (ExpertsPerLane - 1)) == 0, "the fallback sort needs a power of 2");

    __shared__ uint64_t smemCandidateKey[NoAuxTcWarpsPerCta][WARP_SIZE];
    __shared__ float smemCandidateScore[NoAuxTcWarpsPerCta][WARP_SIZE];
    __shared__ float smemTopScore[NoAuxTcWarpsPerCta][MaxSupportedTopExperts];
    __shared__ int32_t smemTopExpert[NoAuxTcWarpsPerCta][MaxSupportedTopExperts];

    int32_t const laneIdx = threadIdx.x % WARP_SIZE;
    int32_t const warpIdx = threadIdx.x / WARP_SIZE;
    int64_t const tokenIdx = static_cast<int64_t>(blockIdx.x) * NoAuxTcWarpsPerCta + warpIdx;
    uint32_t const laneMaskLt = (1u << laneIdx) - 1u;
    auto const slotIsValid = [&](int i) { return AllSlotsValid || i * WARP_SIZE + laneIdx < numExperts; };

    // The bias is a weight: read it before waiting on the grid that wrote the scores.
    float bias[ExpertsPerLane];
#pragma unroll
    for (int i = 0; i < ExpertsPerLane; ++i)
    {
        bias[i] = slotIsValid(i) ? static_cast<float>(routingBias[i * WARP_SIZE + laneIdx]) : 0.F;
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    if (tokenIdx < numTokens)
    {
        InputT const* tokenScores = scores + tokenIdx * numExperts;
        float score[ExpertsPerLane];
#pragma unroll
        for (int i = 0; i < ExpertsPerLane; ++i)
        {
            score[i] = slotIsValid(i) ? static_cast<float>(tokenScores[i * WARP_SIZE + laneIdx]) : 0.F;
        }
        float scoreSigmoid[ExpertsPerLane];
        uint64_t key[ExpertsPerLane];
#pragma unroll
        for (int i = 0; i < ExpertsPerLane; ++i)
        {
            float const sigmoid = sigmoid_accurate(score[i]);
            scoreSigmoid[i] = slotIsValid(i) ? sigmoid : 0.F;
            key[i] = packScoreExpert(slotIsValid(i) ? sigmoid + bias[i] : float{-INFINITY}, i * WARP_SIZE + laneIdx);
        }

        uint64_t laneMax = key[0];
#pragma unroll
        for (int i = 1; i < ExpertsPerLane; ++i)
        {
            laneMax = key[i] > laneMax ? key[i] : laneMax;
        }
        uint32_t const laneMaxHi = static_cast<uint32_t>(laneMax >> 32);
        int32_t reach = 0;
#pragma unroll
        for (int lane = 0; lane < WARP_SIZE; ++lane)
        {
            reach += __shfl_sync(0xffffffff, laneMaxHi, lane) >= laneMaxHi ? 1 : 0;
        }
        uint32_t const tau = warpReduceMax(reach >= topk ? laneMaxHi : 0u);

        int32_t numCandidates = 0;
#pragma unroll
        for (int i = 0; i < ExpertsPerLane; ++i)
        {
            bool const isCandidate = static_cast<uint32_t>(key[i] >> 32) >= tau;
            uint32_t const ballot = __ballot_sync(0xffffffff, isCandidate);
            int32_t const pos = numCandidates + __popc(ballot & laneMaskLt);
            if (isCandidate && pos < WARP_SIZE)
            {
                smemCandidateKey[warpIdx][pos] = key[i];
                smemCandidateScore[warpIdx][pos] = scoreSigmoid[i];
            }
            numCandidates += __popc(ballot);
        }
        __syncwarp();

        if (numCandidates <= WARP_SIZE)
        {
            // Key 0 is below every expert's key (its low half is 65535 - expert > 0), so empty lanes rank nothing.
            uint64_t const candidateKey = laneIdx < numCandidates ? smemCandidateKey[warpIdx][laneIdx] : 0;
            float const candidateScore = laneIdx < numCandidates ? smemCandidateScore[warpIdx][laneIdx] : 0.F;
            int32_t above[4] = {0, 0, 0, 0};
            if (numCandidates <= WARP_SIZE / 2)
            {
#pragma unroll
                for (int lane = 0; lane < WARP_SIZE / 2; ++lane)
                {
                    above[lane % 4] += __shfl_sync(0xffffffff, candidateKey, lane) > candidateKey ? 1 : 0;
                }
            }
            else
            {
#pragma unroll
                for (int lane = 0; lane < WARP_SIZE; ++lane)
                {
                    above[lane % 4] += __shfl_sync(0xffffffff, candidateKey, lane) > candidateKey ? 1 : 0;
                }
            }
            int32_t const rank = (above[0] + above[1]) + (above[2] + above[3]);
            if (laneIdx < numCandidates && rank < topk)
            {
                smemTopScore[warpIdx][rank] = candidateScore;
                smemTopExpert[warpIdx][rank] = unpackExpert(candidateKey);
            }
        }
        else
        {
            // Bitonic sort, descending, then one pick per round: the largest lane head, the lower expert on a tie.
#pragma unroll
            for (int size = 2; size <= ExpertsPerLane; size *= 2)
            {
#pragma unroll
                for (int stride = size / 2; stride > 0; stride /= 2)
                {
#pragma unroll
                    for (int i = 0; i < ExpertsPerLane; ++i)
                    {
                        int const j = i ^ stride;
                        if (j > i)
                        {
                            uint64_t const a = key[i];
                            uint64_t const b = key[j];
                            bool const swap = ((i & size) == 0) ? (a < b) : (a > b);
                            key[i] = swap ? b : a;
                            key[j] = swap ? a : b;
                        }
                    }
                }
            }
            for (int kk = 0; kk < topk; ++kk)
            {
                // Both reductions run on every lane: a full-mask redux that only some lanes reach never returns.
                uint32_t const headHi = static_cast<uint32_t>(key[0] >> 32);
                uint32_t const headLo = static_cast<uint32_t>(key[0]);
                bool const isMax = headHi == warpReduceMax(headHi);
                uint32_t const maxLo = warpReduceMax(isMax ? headLo : 0u);
                if (isMax && headLo == maxLo)
                {
                    int32_t const expert = unpackExpert(key[0]);
                    int32_t const slot = expert / WARP_SIZE;
                    float picked = scoreSigmoid[0];
#pragma unroll
                    for (int i = 1; i < ExpertsPerLane; ++i)
                    {
                        picked = slot == i ? scoreSigmoid[i] : picked;
                    }
                    smemTopScore[warpIdx][kk] = picked;
                    smemTopExpert[warpIdx][kk] = expert;
#pragma unroll
                    for (int i = 0; i < ExpertsPerLane - 1; ++i)
                    {
                        key[i] = key[i + 1];
                    }
                    key[ExpertsPerLane - 1] = packScoreExpert(float{-INFINITY}, 0);
                }
            }
        }
        __syncwarp();

        float const topScore = laneIdx < topk ? smemTopScore[warpIdx][laneIdx] : 0.F;
        int32_t const topExpert = laneIdx < topk ? smemTopExpert[warpIdx][laneIdx] : 0;
        // The picks' sum in cg::reduce's order (an xor butterfly at strides 16, 8, 4, 2, 1), as the block kernel
        // sums them.
        float sum = topScore;
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        {
            sum = sum + __shfl_xor_sync(0xffffffff, sum, offset);
        }
        if (laneIdx < topk)
        {
            OutputT value;
            if constexpr (std::is_same_v<OutputT, float>)
            {
                value = noAuxTcRoutedWeight(topScore, sum, scaleHi, scaleLo, routedScalingFactor);
            }
            else
            {
                value = static_cast<OutputT>(topScore * routedScalingFactor / (sum + 1e-20));
            }
            topkValues[tokenIdx * topk + laneIdx] = value;
            topkIndices[tokenIdx * topk + laneIdx] = static_cast<IdxT>(topExpert);
        }
    }
}

template <typename InputT, typename BiasT, typename OutputT, typename IdxT, int ExpertsPerLane>
static void launchNoAuxTcWarpPerToken(InputT* scores, BiasT* bias, OutputT* topk_values, IdxT* topk_indices,
    int64_t const num_tokens, int64_t const num_experts, int64_t const topk, double const routed_scaling_factor,
    cudaStream_t const stream)
{
    float const scale_hi = static_cast<float>(routed_scaling_factor);
    float const scale_lo = static_cast<float>(routed_scaling_factor - static_cast<double>(scale_hi));

    cudaLaunchConfig_t config;
    config.gridDim = static_cast<unsigned>((num_tokens + NoAuxTcWarpsPerCta - 1) / NoAuxTcWarpsPerCta);
    config.blockDim = NoAuxTcWarpsPerCta * WARP_SIZE;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;

    auto* kernel = num_experts == ExpertsPerLane * WARP_SIZE
        ? &noaux_tc_warp_per_token_kernel<InputT, BiasT, OutputT, IdxT, ExpertsPerLane, true>
        : &noaux_tc_warp_per_token_kernel<InputT, BiasT, OutputT, IdxT, ExpertsPerLane, false>;
    cudaLaunchKernelEx(&config, kernel, static_cast<InputT const*>(scores), static_cast<BiasT const*>(bias),
        topk_values, topk_indices, num_tokens, static_cast<int32_t>(num_experts), static_cast<int32_t>(topk),
        routed_scaling_factor, scale_hi, scale_lo);
}

template <typename InputT, typename BiasT, typename OutputT, typename IdxT>
void invokeNoAuxTc(InputT* scores, BiasT* bias, OutputT* topk_values, IdxT* topk_indices, int64_t const num_tokens,
    int64_t const num_experts, int64_t const n_group, int64_t const topk_group, int64_t const topk,
    double const routed_scaling_factor, cudaStream_t const stream)
{
    bool const is_single_group
        = (n_group <= 1) && (num_experts <= MaxSupportedExpertCount) && (topk <= MaxSupportedTopExperts);

    int64_t const experts_per_group = num_experts / n_group;
    bool const is_multi_group = (n_group > 1) && (num_experts <= NumDeepseekExperts) && (experts_per_group <= WARP_SIZE)
        && (topk <= DefaultMaxNumTopExperts) && (experts_per_group * topk_group <= LargeMaxNumTopGroups * WARP_SIZE);

    if (is_single_group)
    {
        if (num_tokens == 0)
        {
            return;
        }
        if (num_experts <= 4 * WARP_SIZE)
        {
            launchNoAuxTcWarpPerToken<InputT, BiasT, OutputT, IdxT, 4>(
                scores, bias, topk_values, topk_indices, num_tokens, num_experts, topk, routed_scaling_factor, stream);
        }
        else if (num_experts <= 8 * WARP_SIZE)
        {
            launchNoAuxTcWarpPerToken<InputT, BiasT, OutputT, IdxT, 8>(
                scores, bias, topk_values, topk_indices, num_tokens, num_experts, topk, routed_scaling_factor, stream);
        }
        else if (num_experts <= 16 * WARP_SIZE)
        {
            launchNoAuxTcWarpPerToken<InputT, BiasT, OutputT, IdxT, 16>(
                scores, bias, topk_values, topk_indices, num_tokens, num_experts, topk, routed_scaling_factor, stream);
        }
        else
        {
            launchNoAuxTcWarpPerToken<InputT, BiasT, OutputT, IdxT, 32>(
                scores, bias, topk_values, topk_indices, num_tokens, num_experts, topk, routed_scaling_factor, stream);
        }
        sync_check_cuda_error(stream);
    }
    else if (is_multi_group)
    {
        if (num_tokens == 0)
        {
            return;
        }

        cudaLaunchConfig_t config;
        auto* kernel_instance = &deepseek_v3_topk_kernel<InputT, BiasT, OutputT, IdxT, NumDeepseekExperts, true>;
        int num_threads = NumDeepseekExperts;
        if (experts_per_group * topk_group > DefaultMaxNumTopGroups * WARP_SIZE)
        {
            kernel_instance = &deepseek_v3_topk_kernel<InputT, BiasT, OutputT, IdxT, NumDeepseekExperts, true,
                DefaultMaxNumTopExperts, LargeMaxNumTopGroups>;
        }

        config.gridDim = num_tokens;
        config.blockDim = num_threads;
        config.dynamicSmemBytes = 0;
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
        config.numAttrs = 1;
        config.attrs = attrs;

        cudaLaunchKernelEx(&config, kernel_instance, scores, topk_values, topk_indices, bias, num_tokens, n_group,
            topk_group, topk, num_experts, num_experts / n_group, routed_scaling_factor);
        sync_check_cuda_error(stream);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false,
            "invokeNoAuxTc: unsupported configuration (n_group=%ld, num_experts=%ld, topk_group=%ld, topk=%ld). "
            "Please use original pytorch implementation.",
            n_group, num_experts, topk_group, topk);
    }
}

#define INSTANTIATE_NOAUX_TC(InputT, BiasT, OutputT, IdxT)                                                             \
    template void invokeNoAuxTc<InputT, BiasT, OutputT, IdxT>(InputT * scores, BiasT * bias, OutputT * topk_values,    \
        IdxT * topk_indices, int64_t const num_tokens, int64_t const num_experts, int64_t const n_group,               \
        int64_t const topk_group, int64_t const topk, double const routed_scaling_factor, cudaStream_t const stream);

INSTANTIATE_NOAUX_TC(float, float, float, int32_t);
INSTANTIATE_NOAUX_TC(float, half, float, int32_t);

INSTANTIATE_NOAUX_TC(half, float, half, int32_t);
INSTANTIATE_NOAUX_TC(half, half, half, int32_t);

#ifdef ENABLE_BF16
INSTANTIATE_NOAUX_TC(float, __nv_bfloat16, float, int32_t);
INSTANTIATE_NOAUX_TC(half, __nv_bfloat16, half, int32_t);

INSTANTIATE_NOAUX_TC(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t);
INSTANTIATE_NOAUX_TC(__nv_bfloat16, float, __nv_bfloat16, int32_t);
INSTANTIATE_NOAUX_TC(__nv_bfloat16, half, __nv_bfloat16, int32_t);
#endif

} // namespace kernels

TRTLLM_NAMESPACE_END
