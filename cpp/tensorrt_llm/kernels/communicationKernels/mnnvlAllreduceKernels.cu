/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "mnnvlAllreduceKernels.h"
#include "tensorrt_llm/common/config.h"
#include <cooperative_groups.h>
#include <cstddef>
#include <cstdint>
#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/lamportUtils.cuh"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/common/tllmDataType.h"
#include "tensorrt_llm/kernels/quantization.cuh"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::mnnvl
{

using tensorrt_llm::common::isNegZero;
using tensorrt_llm::common::isLamportDirty;
using tensorrt_llm::common::LamportFlags;
using tensorrt_llm::common::cuda_cast;
using tensorrt_llm::common::getMultiProcessorCount;
using tensorrt_llm::common::getDTypeSize;
using tensorrt_llm::common::loadPackedVolatile;

// Guard the helper function used for this kernel.
namespace detail
{
template <typename PackedType, typename T>
union PackedVec
{
    PackedType packed;
    T elements[sizeof(PackedType) / sizeof(T)];

    __device__ PackedVec& operator+=(PackedVec& other)
    {
#pragma unroll
        for (int i = 0; i < sizeof(PackedType) / sizeof(T); i++)
        {
            elements[i] += other.elements[i];
        }
        return *this;
    }

    __device__ PackedVec operator+(PackedVec& other)
    {
        PackedVec result;
#pragma unroll
        for (int i = 0; i < sizeof(PackedType) / sizeof(T); i++)
        {
            result.elements[i] = elements[i] + other.elements[i];
        }
        return result;
    }
};

template <typename PackedType, typename T>
inline __device__ PackedType loadPacked(T* ptr)
{
    return *reinterpret_cast<PackedType*>(ptr);
}

template <typename PackedType, typename T>
inline __device__ const PackedType loadPacked(T const* ptr)
{
    return *reinterpret_cast<PackedType const*>(ptr);
}

template <typename T_IN>
inline __device__ void copyF4(T_IN* dst, T_IN const* src)
{
    float4* dst4 = reinterpret_cast<float4*>(dst);
    float4 const* src4 = reinterpret_cast<float4 const*>(src);
    __pipeline_memcpy_async(dst4, src4, sizeof(float4));
}

uint32_t constexpr kWARP_SIZE = 32U;
uint32_t constexpr kLOG2_WARP_SIZE = 5U;
uint32_t constexpr kLANE_ID_MASK = 0x1f;

template <typename T>
inline __device__ T warpReduceSumPartial(T val)
{
    int laneId = threadIdx.x & kLANE_ID_MASK;
    // We make sure only the last warp will call this function
    int warpSize = blockDim.x - (threadIdx.x & ~(kWARP_SIZE - 1));
    unsigned int active_mask = (1U << warpSize) - 1;

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        int targetLane = laneId ^ mask;
        auto tmp = __shfl_xor_sync(active_mask, val, mask, kWARP_SIZE);
        val += targetLane < warpSize ? tmp : 0;
    }
    return val;
}

// SYNC:
//  - True: share the sume across all threads
//  - False: only thread 0 get the sum; Other thread's value is undefined.
template <typename T, bool SYNC = false>
inline __device__ T blockReduceSumPartial(T val)
{
    __shared__ T smem[kWARP_SIZE];
    int laneId = threadIdx.x & kLANE_ID_MASK;
    int warpId = threadIdx.x >> kLOG2_WARP_SIZE;
    int warpNum = (blockDim.x + kWARP_SIZE - 1) >> kLOG2_WARP_SIZE; // Ceiling division to include partial warps

    val = (warpId == warpNum - 1) ? warpReduceSumPartial(val) : tensorrt_llm::common::warpReduceSum(val);
    if (laneId == 0)
    {
        smem[warpId] = val;
    }
    __syncthreads();

    if (warpId == 0)
    {
        val = (laneId < warpNum) ? smem[laneId] : (T) 0.f;
        // Need to consider the corner case where we only have one warp and it is partial
        val = (warpNum == 1) ? warpReduceSumPartial(val) : tensorrt_llm::common::warpReduceSum(val);

        if constexpr (SYNC)
        {
            if (laneId == 0)
            {
                smem[warpId] = val;
            }
        }
    }
    if constexpr (SYNC)
    {
        __syncthreads();
        val = smem[0];
    }
    return val;
}

// blockReduceSum in reduceKernelUtils.cuh returns result only on warp0
// So we need a duplicate implementation here where all threads get the result
template <typename T>
inline __device__ T blockReduceSumFull(T val)
{
    __shared__ T smem[kWARP_SIZE];
    int lane_id = threadIdx.x & kLANE_ID_MASK;
    int warp_id = threadIdx.x >> kLOG2_WARP_SIZE;
    int warp_num = blockDim.x >> kLOG2_WARP_SIZE;

    val = tensorrt_llm::common::warpReduceSum(val);
    if (lane_id == 0)
    {
        smem[warp_id] = val;
    }
    __syncthreads();

    val = (lane_id < warp_num) ? smem[lane_id] : (T) 0.f;
    val = tensorrt_llm::common::warpReduceSum(val);

    return val;
}

template <typename T, bool SYNC = false>
inline __device__ T blockReduceSum(T val)
{
    bool hasPartialWarp = (blockDim.x & kLANE_ID_MASK) != 0;
    if (hasPartialWarp)
    {
        return blockReduceSumPartial<T, SYNC>(val);
    }
    else
    {
        return blockReduceSumFull<T>(val);
    }
}

// We have to define this again since the one in mathUtils.h is shadowed by the one from cudaUtils.h, which is a
// host-only function!
template <typename T>
inline __device__ __host__ T divUp(T m, T n)
{
    return (m + n - 1) / n;
}

inline bool requiresTwoAccessScaleGroup(ar_fusion::AllReduceFusionPattern pattern)
{
    return pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNormFP4Quant
        || pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP4Quant;
}

// A helper function to tune the grid configuration for fused oneshot and RMSNorm kernels.
// Return (block_size, cluster_size, loads_per_thread).
template <bool UseCluster>
std::tuple<int, int, int> adjustGridConfig(int numTokens, int dim, int eltsPerThread, int accessGroupSize = 1)
{
    // Step 1: Start from the widest cluster we are willing to launch. The caller can request a no-CGA
    // fallback for multi-load RMSNorm rows, in which case the cluster width stays at one CTA.
    int clusterSize = UseCluster ? 8 : 1;
    int blockSize = 128;
    int threadsNeeded = divUp(dim, eltsPerThread);
    int loadsPerThread = 1;

    blockSize = divUp(threadsNeeded, clusterSize);
    if constexpr (UseCluster)
    {
        // Step 2: Shrink the cluster until the hidden dimension partitions cleanly across CTAs
        // and each CTA contains an integral packed-access group. NVFP4 scale generation uses a
        // two-access group, so it must not straddle a CTA boundary.
        while ((threadsNeeded % clusterSize != 0 || divUp(threadsNeeded, clusterSize) % accessGroupSize != 0)
            && clusterSize > 1)
        {
            clusterSize /= 2;
        }
        int const maxDivisibleClusterSize = clusterSize;
        blockSize = divUp(threadsNeeded, clusterSize);
        // Step 3: If divisibility leaves each CTA too small, trade cluster width for at least
        // a 128-thread CTA. This improves occupancy and avoids tiny CTAs when the row is narrow.
        while (blockSize < 128 && clusterSize >= 2)
        {
            blockSize *= 2;
            clusterSize /= 2;
        }
        int smCount = getMultiProcessorCount();
        // Step 4: If the token grid already has enough CTAs to cover the GPU, reduce cluster
        // width and make CTAs larger. This avoids over-partitioning one token across too many CTAs.
        while (numTokens * clusterSize > smCount && clusterSize > 1 && blockSize <= 512)
        {
            blockSize *= 2;
            clusterSize /= 2;
        }
        // Step 5: If the token grid still underfills the GPU, restore cluster width up to the
        // divisibility limit. We accept 64-thread CTAs here to expose more CTAs per token.
        while (clusterSize < maxDivisibleClusterSize)
        {
            int const candidateClusterSize = clusterSize * 2;
            int const candidateBlockSize = divUp(threadsNeeded, candidateClusterSize);
            if (candidateBlockSize < 64 || candidateBlockSize % accessGroupSize != 0
                || numTokens * candidateClusterSize > smCount)
            {
                break;
            }
            clusterSize = candidateClusterSize;
            blockSize = candidateBlockSize;
        }
    }

    // Step 6: For very wide rows, first increase cluster width on SM90+ and then increase
    // per-thread loads. The goal is to keep block_size within CUDA's 1024-thread limit.
    while (blockSize > 1024)
    {
        if constexpr (UseCluster)
        {
            if (clusterSize < 8)
            {
                clusterSize = clusterSize << 1;
            }
            else
            {
                if (loadsPerThread < 8)
                {
                    loadsPerThread += 1;
                }
                else
                {
                    break;
                }
            }
        }
        else
        {
            if (loadsPerThread < 8)
            {
                loadsPerThread += 1;
            }
            else
            {
                break;
            }
        }
        blockSize = divUp(threadsNeeded, clusterSize * loadsPerThread);
    }

    while (blockSize % accessGroupSize != 0 && loadsPerThread < 8)
    {
        loadsPerThread += 1;
        blockSize = divUp(threadsNeeded, clusterSize * loadsPerThread);
    }

    return {blockSize, clusterSize, loadsPerThread};
}

template <typename T>
struct MnnvlAllReduceKernelParams
{
    T* outputPtr;
    T* residualOutPtr;
    T const* shardPtr;
    T const* residualInPtr;
    T const* gammaPtr;
    T** inputPtrs;
    T* bufferInputPtr;
    T* mcastPtr;
    void* quantOutPtr;
    void* scaleOutPtr;
    float const* scaleFactorPtr;
    int numTokens;
    int tokenDim;
    int nRanks;
    int rank;
    float epsilon;
    uint32_t* bufferFlags;
    bool waitForResults;
    QuantizationSFLayout layout;
};

template <typename PackedType, typename T>
inline __device__ void sanitizeLamportPayload(PackedVec<PackedType, T>& value)
{
#pragma unroll
    for (int i = 0; i < sizeof(PackedType) / sizeof(T); i++)
    {
        if (isNegZero(value.elements[i]))
        {
            value.elements[i] = cuda_cast<T, float>(0.F);
        }
    }
}

template <int Rank, uint8_t WorldSize, uint8_t LocalRank, typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ bool pollOneshotRemoteRank(
    PackedVec<PackedType, T>* remoteValues, T* stagePtrLocal, int token, int tokenDim, int packedIdx)
{
    if constexpr (Rank == LocalRank)
    {
        return true;
    }
    else
    {
        auto loaded = loadPackedVolatile<PackedType>(
            &stagePtrLocal[token * tokenDim * WorldSize + Rank * tokenDim + packedIdx * kELTS_PER_THREAD]);
        remoteValues[Rank].packed = loaded.packed;
        return !isLamportDirty(loaded);
    }
}

template <uint8_t WorldSize, uint8_t LocalRank, typename T, typename PackedType, int kELTS_PER_THREAD, int... Ranks>
inline __device__ bool pollOneshotRemoteRanks(PackedVec<PackedType, T>* remoteValues, T* stagePtrLocal, int token,
    int tokenDim, int packedIdx, std::integer_sequence<int, Ranks...>)
{
    bool valid = true;
    ((valid &= pollOneshotRemoteRank<Ranks, WorldSize, LocalRank, T, PackedType, kELTS_PER_THREAD>(
          remoteValues, stagePtrLocal, token, tokenDim, packedIdx)),
        ...);
    return valid;
}

template <typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ void accumulatePacked(float (&accum)[kELTS_PER_THREAD], PackedVec<PackedType, T> const& value)
{
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++)
    {
        accum[i] += cuda_cast<float, T>(value.elements[i]);
    }
}

template <uint8_t WorldSize, int kRankChunk, typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ void accumulateLamportRanksChunked(
    float (&accum)[kELTS_PER_THREAD], T const* input, int token, int tokenDim, int packedIdx)
{
    static_assert(kRankChunk > 0);
    static_assert(WorldSize % kRankChunk == 0);

#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++)
    {
        accum[i] = 0.F;
    }

#pragma unroll 1
    for (int rankBase = 0; rankBase < WorldSize; rankBase += kRankChunk)
    {
        float chunkAccum[kELTS_PER_THREAD];
        while (1)
        {
            bool valid = true;
#pragma unroll
            for (int i = 0; i < kELTS_PER_THREAD; i++)
            {
                chunkAccum[i] = 0.F;
            }
#pragma unroll
            for (int rr = 0; rr < kRankChunk; rr++)
            {
                int const r = rankBase + rr;
                auto loaded = loadPackedVolatile<PackedType>(
                    &input[token * tokenDim * WorldSize + r * tokenDim + packedIdx * kELTS_PER_THREAD]);
                PackedVec<PackedType, T> value;
                value.packed = loaded.packed;
                valid &= !isLamportDirty(loaded);
                accumulatePacked<T, PackedType, kELTS_PER_THREAD>(chunkAccum, value);
            }
            if (valid)
            {
                break;
            }
        }
#pragma unroll
        for (int i = 0; i < kELTS_PER_THREAD; i++)
        {
            accum[i] += chunkAccum[i];
        }
    }
}

template <int Rank, uint8_t LocalRank, typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ void accumulateOneshotRank(float (&accum)[kELTS_PER_THREAD],
    PackedVec<PackedType, T> const* remoteValues, PackedVec<PackedType, T> const& localValue)
{
    if constexpr (Rank == LocalRank)
    {
        accumulatePacked<T, PackedType, kELTS_PER_THREAD>(accum, localValue);
    }
    else
    {
        accumulatePacked<T, PackedType, kELTS_PER_THREAD>(accum, remoteValues[Rank]);
    }
}

template <uint8_t LocalRank, typename T, typename PackedType, int kELTS_PER_THREAD, int... Ranks>
inline __device__ void accumulateOneshotRanks(float (&accum)[kELTS_PER_THREAD],
    PackedVec<PackedType, T> const* remoteValues, PackedVec<PackedType, T> const& localValue,
    std::integer_sequence<int, Ranks...>)
{
    (accumulateOneshotRank<Ranks, LocalRank, T, PackedType, kELTS_PER_THREAD>(accum, remoteValues, localValue), ...);
}

template <uint8_t WorldSize, uint8_t LocalRank, typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ void waitOneshotRemoteRanks(
    PackedVec<PackedType, T>* remoteValues, T* stagePtrLocal, int token, int tokenDim, int packedIdx)
{
    static_assert(LocalRank < WorldSize);
    while (1)
    {
        bool const valid = pollOneshotRemoteRanks<WorldSize, LocalRank, T, PackedType, kELTS_PER_THREAD>(
            remoteValues, stagePtrLocal, token, tokenDim, packedIdx, std::make_integer_sequence<int, WorldSize>{});
        if (valid)
        {
            break;
        }
    }
}

template <uint8_t WorldSize, uint8_t LocalRank, typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ PackedVec<PackedType, T> reduceOneshotDeterministic(
    PackedVec<PackedType, T> const* remoteValues, PackedVec<PackedType, T> const& localValue)
{
    static_assert(LocalRank < WorldSize);
    float accum[kELTS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++)
    {
        accum[i] = 0.F;
    }
    accumulateOneshotRanks<LocalRank, T, PackedType, kELTS_PER_THREAD>(
        accum, remoteValues, localValue, std::make_integer_sequence<int, WorldSize>{});

    PackedVec<PackedType, T> packedAccum;
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++)
    {
        packedAccum.elements[i] = cuda_cast<T, float>(accum[i]);
    }
    return packedAccum;
}

template <uint8_t WorldSize, uint8_t LocalRank, typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ PackedVec<PackedType, T> reduceOneshotDeterministicFastPath(
    PackedVec<PackedType, T> const& localValue, T* stagePtrLocal, int token, int tokenDim, int packedIdx)
{
    PackedVec<PackedType, T> remoteValues[WorldSize];
    waitOneshotRemoteRanks<WorldSize, LocalRank, T, PackedType, kELTS_PER_THREAD>(
        remoteValues, stagePtrLocal, token, tokenDim, packedIdx);
    return reduceOneshotDeterministic<WorldSize, LocalRank, T, PackedType, kELTS_PER_THREAD>(remoteValues, localValue);
}

template <ar_fusion::AllReduceFusionPattern Pattern, typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ void quantizeEpilogue(PackedVec<PackedType, T> const& value,
    MnnvlAllReduceKernelParams<T> const& params, int packedAccessIdx, int accessIdInToken, int token)
{
    if constexpr (ar_fusion::GetQuantType<Pattern> == ar_fusion::QuantType::kFP4)
    {
        static_assert(kELTS_PER_THREAD == 8, "NVFP4 quantization expects eight elements per 16-byte access.");
        constexpr int kSFVecSize = 16;
        using QuantPackedVec = tensorrt_llm::kernels::PackedVec<T>;
        QuantPackedVec packVal = *reinterpret_cast<QuantPackedVec const*>(&value.packed);
        auto* sfOut = cvt_quant_get_sf_out_offset<uint32_t, 2>(std::nullopt /* batchIdx */, token, accessIdInToken,
            std::nullopt /* numRows */, params.tokenDim / kSFVecSize, reinterpret_cast<uint32_t*>(params.scaleOutPtr),
            params.layout);
        reinterpret_cast<uint32_t*>(params.quantOutPtr)[packedAccessIdx]
            = cvt_warp_fp16_to_fp4<T, kSFVecSize, false>(packVal, *params.scaleFactorPtr, sfOut);
    }
    else if constexpr (ar_fusion::GetQuantType<Pattern> == ar_fusion::QuantType::kFP8)
    {
        float const scale = 1.F / *params.scaleFactorPtr;
        using PackedQuantizedType = std::conditional_t<std::is_same_v<T, float>, float, float2>;
        PackedQuantizedType quantized;
#pragma unroll
        for (int i = 0; i < kELTS_PER_THREAD; ++i)
        {
            reinterpret_cast<__nv_fp8_e4m3*>(&quantized)[i]
                = static_cast<__nv_fp8_e4m3>(static_cast<float>(value.elements[i]) * scale);
        }
        reinterpret_cast<PackedQuantizedType*>(params.quantOutPtr)[packedAccessIdx] = quantized;
    }
    else
    {
        static_assert(ar_fusion::GetQuantType<Pattern> == ar_fusion::QuantType::kNone, "Invalid quant type");
    }
}

template <ar_fusion::AllReduceFusionPattern Pattern, typename T, typename PackedType, int kELTS_PER_THREAD>
inline __device__ void writeEpilogueOutput(PackedVec<PackedType, T> const& value,
    MnnvlAllReduceKernelParams<T> const& params, int threadOffset, int packedAccessIdx, int accessIdInToken, int token)
{
    if constexpr (ar_fusion::HasAllReduceOut<Pattern> || ar_fusion::HasNormOut<Pattern>)
    {
        if (params.outputPtr != nullptr)
        {
            reinterpret_cast<PackedType*>(&params.outputPtr[threadOffset])[0] = value.packed;
        }
    }
    if constexpr (ar_fusion::GetQuantType<Pattern> != ar_fusion::QuantType::kNone)
    {
        quantizeEpilogue<Pattern, T, PackedType, kELTS_PER_THREAD>(
            value, params, packedAccessIdx, accessIdInToken, token);
    }
}

} // namespace detail

using detail::PackedVec;
using detail::loadPacked;
using detail::blockReduceSum;
using detail::divUp;
using detail::copyF4;
using detail::MnnvlAllReduceKernelParams;
using detail::sanitizeLamportPayload;
using detail::accumulateLamportRanksChunked;
using detail::reduceOneshotDeterministicFastPath;
using detail::writeEpilogueOutput;

template <uint8_t WorldSize, typename T,
    ar_fusion::AllReduceFusionPattern Pattern = ar_fusion::AllReduceFusionPattern::kAllReduce,
    typename PackedType = float4>
__global__ void __launch_bounds__(1024) oneshotAllreduceFusionKernel(MnnvlAllReduceKernelParams<T> params)
{
    constexpr int kELTS_PER_THREAD = sizeof(PackedType) / sizeof(T);
    constexpr uint32_t kELT_SIZE = sizeof(T);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    int packedIdx = cluster.thread_rank();
    int token = blockIdx.x;
    int threadOffset = token * params.tokenDim + packedIdx * kELTS_PER_THREAD;

    cudaGridDependencySynchronize();
#else
    int packedIdx = blockIdx.y * blockDim.x + threadIdx.x;
    int token = blockIdx.x;
    // Offset w.r.t. the input shard
    int threadOffset = token * params.tokenDim + packedIdx * kELTS_PER_THREAD;
#endif

    // We only use 1 stage for the oneshot allreduce
    LamportFlags<PackedType, true> flag(params.bufferFlags, 1);
    T* stagePtrMcast = reinterpret_cast<T*>(flag.getCurLamportBuf(params.mcastPtr, 0));
    T* stagePtrLocal = reinterpret_cast<T*>(flag.getCurLamportBuf(params.inputPtrs[params.rank], 0));
    bool const inBounds = packedIdx * kELTS_PER_THREAD < params.tokenDim;

    // ==================== Broadcast tokens to each rank =============================
    PackedVec<PackedType, T> val;
    if (inBounds)
    {
        val.packed = loadPacked<PackedType>(&params.shardPtr[threadOffset]);
        sanitizeLamportPayload(val);

        reinterpret_cast<PackedType*>(
            &stagePtrMcast[token * params.tokenDim * WorldSize + params.rank * params.tokenDim])[packedIdx]
            = val.packed;
    }

    flag.ctaArrive();
    // ======================= Lamport Sync and clear the output buffer from previous iteration
    // =============================
    flag.clearDirtyLamportBuf(params.inputPtrs[params.rank], -1);

    if (!inBounds)
    {
        return;
    }

    // ======================= Reduction =============================
    // Fully deterministic: every rank uses the exact same reduction order. For WorldSize <= 8, specialize the local
    // slot so the fast path reuses `val` from registers without a dynamic `remoteValues[params.rank]` store. Larger
    // world sizes use the compact fallback because the benefit is thin but specializing every rank significantly
    // increases compile time.
    PackedVec<PackedType, T> packedAccum;
    if constexpr (WorldSize <= 8)
    {
        packedAccum = val;
#define RUN_ONESHOT_LOCAL_RANK(LOCAL_RANK)                                                                             \
    case LOCAL_RANK:                                                                                                   \
        if constexpr (WorldSize > LOCAL_RANK)                                                                          \
        {                                                                                                              \
            packedAccum = reduceOneshotDeterministicFastPath<WorldSize, LOCAL_RANK, T, PackedType, kELTS_PER_THREAD>(  \
                val, stagePtrLocal, token, params.tokenDim, packedIdx);                                                \
        }                                                                                                              \
        break

        switch (params.rank)
        {
            RUN_ONESHOT_LOCAL_RANK(0);
            RUN_ONESHOT_LOCAL_RANK(1);
            RUN_ONESHOT_LOCAL_RANK(2);
            RUN_ONESHOT_LOCAL_RANK(3);
            RUN_ONESHOT_LOCAL_RANK(4);
            RUN_ONESHOT_LOCAL_RANK(5);
            RUN_ONESHOT_LOCAL_RANK(6);
            RUN_ONESHOT_LOCAL_RANK(7);
        }
#undef RUN_ONESHOT_LOCAL_RANK
    }
    else
    {
        // Chunk Lamport polling so only a bounded rank set is live at once, avoiding register spills for large
        // world sizes.
        constexpr int kRankChunk = 8;
        float accum[kELTS_PER_THREAD];
        accumulateLamportRanksChunked<WorldSize, kRankChunk, T, PackedType, kELTS_PER_THREAD>(
            accum, stagePtrLocal, token, params.tokenDim, packedIdx);
#pragma unroll
        for (int i = 0; i < kELTS_PER_THREAD; i++)
        {
            packedAccum.elements[i] = cuda_cast<T, float>(accum[i]);
        }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    if constexpr (ar_fusion::HasResidual<Pattern>)
    {
        // =============================== Residual ===============================
        PackedVec<PackedType, T> residualIn;
        residualIn.packed = *reinterpret_cast<PackedType const*>(&params.residualInPtr[threadOffset]);
        packedAccum += residualIn;
        if constexpr (ar_fusion::HasResidualOut<Pattern>)
        {
            *reinterpret_cast<PackedType*>(&params.residualOutPtr[threadOffset]) = packedAccum.packed;
        }
    }
    if constexpr (ar_fusion::HasRMSNorm<Pattern>)
    {
        // =============================== Rmsnorm ================================
        PackedVec<PackedType, T> gamma;
        gamma.packed = *reinterpret_cast<PackedType const*>(&params.gammaPtr[packedIdx * kELTS_PER_THREAD]);

        float threadSum = 0.F;
#pragma unroll
        for (int i = 0; i < kELTS_PER_THREAD; i++)
        {
            // FIXME: Use float square if accuracy issue
            threadSum += cuda_cast<float, T>(packedAccum.elements[i] * packedAccum.elements[i]);
        }
        float blockSum = blockReduceSum<float, true>(threadSum);

        float fullSum = blockSum;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        __shared__ float sharedVal[8]; // Temporary variable to share the sum within block
        namespace cg = cooperative_groups;
        cg::cluster_group cluster = cg::this_cluster();
        int const numBlocks = cluster.num_blocks();
        if (numBlocks > 1)
        {
            fullSum = 0.F;
            // Need to reduce over the entire cluster
            int const blockRank = cluster.block_rank();
            if (threadIdx.x < numBlocks)
            {
                cluster.map_shared_rank(&sharedVal[0], threadIdx.x)[blockRank] = blockSum;
            }
            // cluster.sync();
            cluster.barrier_wait(cluster.barrier_arrive());
            for (int i = 0; i < numBlocks; ++i)
            {
                fullSum += sharedVal[i];
            }
        }
#endif
        float rcpRms = rsqrtf(fullSum / params.tokenDim + params.epsilon);
#pragma unroll
        for (int i = 0; i < kELTS_PER_THREAD; i++)
        {
            packedAccum.elements[i] = cuda_cast<T, float>(
                cuda_cast<float, T>(packedAccum.elements[i]) * rcpRms * cuda_cast<float, T>(gamma.elements[i]));
        }
    }
    int const packedAccessIdx = threadOffset / kELTS_PER_THREAD;
    writeEpilogueOutput<Pattern, T, PackedType, kELTS_PER_THREAD>(
        packedAccum, params, threadOffset, packedAccessIdx, packedIdx, token);
    flag.waitAndUpdate({static_cast<uint32_t>(params.numTokens * params.tokenDim * WorldSize * kELT_SIZE), 0, 0, 0});
}

using detail::adjustGridConfig;
using detail::requiresTwoAccessScaleGroup;

void oneshotAllreduceFusionOp(AllReduceFusionParams const& params)
{

    static int const kSMVersion = tensorrt_llm::common::getSMVersion();
    TLLM_CHECK_WITH_INFO(kSMVersion >= 90, "[MNNVL AllReduceOneShot] requires SM 90 or newer.");
    int const numTokens = params.numTokens;
    int const tokenDim = params.tokenDim;
    int const eltsPerThread = sizeof(float4) / getDTypeSize(params.dType);
    int const accessGroupSize = requiresTwoAccessScaleGroup(params.pattern) ? 2 : 1;

    auto [blockSize, clusterSize, loadsPerThread]
        = adjustGridConfig<true>(numTokens, tokenDim, eltsPerThread, accessGroupSize);
    dim3 grid(numTokens, clusterSize, 1);

    TLLM_LOG_DEBUG(
        "[MNNVL AllReduceOneShot] Dispatch: grid size: (%d, %d, 1), block_size: %d, cluster_size: %d, "
        "loads_per_thread: %d, deterministic_reduction_order: 1, threads_needed: %d",
        numTokens, clusterSize, blockSize, clusterSize, loadsPerThread, divUp(tokenDim, eltsPerThread));

    TLLM_CHECK_WITH_INFO(blockSize <= 1024 && loadsPerThread == 1,
        "Hidden Dimension %d exceeds the maximum supported hidden dimension (%d)", tokenDim, 1024 * 8 * eltsPerThread);

    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    attrs[1].id = cudaLaunchAttributeClusterDimension;
    attrs[1].val.clusterDim.x = 1;
    attrs[1].val.clusterDim.y = clusterSize;
    attrs[1].val.clusterDim.z = 1;

    cudaLaunchConfig_t config{
        .gridDim = grid,
        .blockDim = blockSize,
        .dynamicSmemBytes = 0,
        .stream = params.stream,
        .attrs = attrs,
        .numAttrs = 2U,
    };

#define LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, T, PATTERN)                                                                \
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, &oneshotAllreduceFusionKernel<WORLD_SIZE, T, PATTERN>, kernelParams));
#define DISPATCH_ALLREDUCE_PATTERN(WORLD_SIZE, T)                                                                      \
    if (params.pattern == ar_fusion::AllReduceFusionPattern::kAllReduce)                                               \
    {                                                                                                                  \
        LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, T, ar_fusion::AllReduceFusionPattern::kAllReduce);                         \
    }                                                                                                                  \
    else if (params.pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNorm)                                  \
    {                                                                                                                  \
        LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, T, ar_fusion::AllReduceFusionPattern::kARResidualRMSNorm);                 \
    }                                                                                                                  \
    else if (params.pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNormFP8Quant)                          \
    {                                                                                                                  \
        LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, T, ar_fusion::AllReduceFusionPattern::kARResidualRMSNormFP8Quant);         \
    }                                                                                                                  \
    else if (params.pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP8Quant)                       \
    {                                                                                                                  \
        LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, T, ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP8Quant);      \
    }                                                                                                                  \
    else if (params.pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNormFP4Quant)                          \
    {                                                                                                                  \
        if constexpr (!std::is_same_v<T, float>)                                                                       \
        {                                                                                                              \
            LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, T, ar_fusion::AllReduceFusionPattern::kARResidualRMSNormFP4Quant);     \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            TLLM_CHECK_WITH_INFO(false, "[MNNVL AllReduceOneShot] NVFP4 quantization does not support FP32 input.");   \
        }                                                                                                              \
    }                                                                                                                  \
    else if (params.pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP4Quant)                       \
    {                                                                                                                  \
        if constexpr (!std::is_same_v<T, float>)                                                                       \
        {                                                                                                              \
            LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, T, ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP4Quant);  \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            TLLM_CHECK_WITH_INFO(false, "[MNNVL AllReduceOneShot] NVFP4 quantization does not support FP32 input.");   \
        }                                                                                                              \
    }                                                                                                                  \
    else if (params.pattern == ar_fusion::AllReduceFusionPattern::kARRMSNorm)                                          \
    {                                                                                                                  \
        LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, T, ar_fusion::AllReduceFusionPattern::kARRMSNorm);                         \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        TLLM_CHECK_WITH_INFO(false, "[MNNVL AllReduceOneShot] Unsupported fusion pattern.");                           \
    }
    // C++17 compatible alternative using a template function
    auto dispatchImpl = [&](auto* type_ptr) -> bool
    {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        T** ucPtrs = reinterpret_cast<T**>(params.bufferPtrsDev);
        T* mcPtr = reinterpret_cast<T*>(params.multicastPtr);
        T* output = reinterpret_cast<T*>(params.output);
        T* residualOut = reinterpret_cast<T*>(params.residualOut);
        T const* input = reinterpret_cast<T const*>(params.input);
        T const* residualIn = reinterpret_cast<T const*>(params.residualIn);
        T const* gamma = reinterpret_cast<T const*>(params.gamma);
        MnnvlAllReduceKernelParams<T> kernelParams{output, residualOut, input, residualIn, gamma, ucPtrs,
            reinterpret_cast<T*>(params.bufferPtrLocal), mcPtr, params.quantOut, params.scaleOut, params.scaleFactor,
            numTokens, tokenDim, params.nRanks, params.rank, static_cast<float>(params.epsilon), params.bufferFlags,
            false, params.layout};

        switch (params.nRanks)
        {
            // FIXME: Do we need other world sizes?
        case 2: DISPATCH_ALLREDUCE_PATTERN(2, T); return true;
        case 4: DISPATCH_ALLREDUCE_PATTERN(4, T); return true;
        case 8: DISPATCH_ALLREDUCE_PATTERN(8, T); return true;
        case 16: DISPATCH_ALLREDUCE_PATTERN(16, T); return true;
        case 32: DISPATCH_ALLREDUCE_PATTERN(32, T); return true;
        case 64: DISPATCH_ALLREDUCE_PATTERN(64, T); return true;
        }
        return false;
    };
#undef LAUNCH_ALLREDUCE_KERNEL
#undef DISPATCH_ALLREDUCE_PATTERN
    bool launched = (params.dType == tensorrt_llm::DataType::kBF16 && dispatchImpl((__nv_bfloat16*) nullptr))
        || (params.dType == tensorrt_llm::DataType::kFLOAT && dispatchImpl((float*) nullptr))
        || (params.dType == tensorrt_llm::DataType::kHALF && dispatchImpl((__nv_half*) nullptr));
    if (!launched)
    {
        TLLM_CHECK_WITH_INFO(false, "Failed to dispatch MNNVL AllReduceOneShot kernel.");
    }
}

enum MNNVLTwoShotStage : uint8_t
{
    SCATTER = 0,
    BROADCAST = 1,
    NUM_STAGES = 2,
};

template <uint8_t WorldSize, typename T, typename PackedType = float4>
__global__ __launch_bounds__(128) void twoshotAllreduceKernel(MnnvlAllReduceKernelParams<T> params)
{
    constexpr int kELTS_PER_THREAD = sizeof(PackedType) / sizeof(T);
    constexpr uint32_t kELT_SIZE = sizeof(T);

    int packedIdx = blockIdx.y * blockDim.x + threadIdx.x;
    int token = blockIdx.x;
    // Offset w.r.t. the input shard
    int threadOffset = token * params.tokenDim + packedIdx * kELTS_PER_THREAD;

    int destRank = token % WorldSize;
    int destTokenOffset = token / WorldSize;
    bool const inBounds = packedIdx * kELTS_PER_THREAD < params.tokenDim;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    LamportFlags<PackedType, false> flag(params.bufferFlags, MNNVLTwoShotStage::NUM_STAGES);

    T* scatterBufLocal
        = reinterpret_cast<T*>(flag.getCurLamportBuf(params.inputPtrs[params.rank], MNNVLTwoShotStage::SCATTER));
    T* scatterBufDest
        = reinterpret_cast<T*>(flag.getCurLamportBuf(params.inputPtrs[destRank], MNNVLTwoShotStage::SCATTER));
    T* broadcastBufW = reinterpret_cast<T*>(flag.getCurLamportBuf(params.mcastPtr, MNNVLTwoShotStage::BROADCAST));
    T* broadcastBufR
        = reinterpret_cast<T*>(flag.getCurLamportBuf(params.inputPtrs[params.rank], MNNVLTwoShotStage::BROADCAST));

    // =============================== Scatter ===============================

    // Load vectorized data
    PackedVec<PackedType, T> val;
    if (inBounds)
    {
        val.packed = loadPacked<PackedType>(&params.shardPtr[threadOffset]);
        sanitizeLamportPayload(val);

        // Store vectorized data
        reinterpret_cast<PackedType*>(
            &scatterBufDest[destTokenOffset * params.tokenDim * WorldSize + params.rank * params.tokenDim])[packedIdx]
            = val.packed;
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif

    flag.clearDirtyLamportBuf(params.inputPtrs[params.rank], MNNVLTwoShotStage::SCATTER);

    // =============================== Reduction and Broadcast ===============================

    if (inBounds && (token % WorldSize) == params.rank)
    {
        int localToken = token / WorldSize;

        // Store vectorized result
        PackedVec<PackedType, T> packedAccum;
        // Chunk Lamport polling so only a bounded rank set is live at once, avoiding register spills for large
        // world sizes.
        constexpr int kRankChunk = WorldSize < 16 ? WorldSize : 16;
        float accum[kELTS_PER_THREAD];
        accumulateLamportRanksChunked<WorldSize, kRankChunk, T, PackedType, kELTS_PER_THREAD>(
            accum, scatterBufLocal, localToken, params.tokenDim, packedIdx);
#pragma unroll
        for (int i = 0; i < kELTS_PER_THREAD; i++)
        {
            packedAccum.elements[i] = cuda_cast<T, float>(accum[i]);
        }
        sanitizeLamportPayload(packedAccum);
        reinterpret_cast<PackedType*>(&broadcastBufW[token * params.tokenDim])[packedIdx] = packedAccum.packed;
    }

    flag.clearDirtyLamportBuf(params.inputPtrs[params.rank], MNNVLTwoShotStage::BROADCAST);

    // Optionally wait for results if the next layer isn't doing the Lamport check
    if (params.waitForResults)
    {
        // Update the atomic counter to indicate the block has read the offsets
        flag.ctaArrive();

        if (inBounds)
        {
            auto loaded = loadPackedVolatile<PackedType>(&broadcastBufR[threadOffset]);
            while (isLamportDirty(loaded))
            {
                loaded = loadPackedVolatile<PackedType>(&broadcastBufR[threadOffset]);
            }
            if (params.outputPtr)
            {
                reinterpret_cast<PackedType*>(&params.outputPtr[threadOffset])[0] = loaded.packed;
            }
        }

        // Update the buffer flags
        flag.waitAndUpdate({static_cast<uint32_t>(divUp<uint32_t>(params.numTokens, WorldSize) * WorldSize
                                * params.tokenDim * kELT_SIZE),                    // Clear Size for scatter stage
            static_cast<uint32_t>(params.numTokens * params.tokenDim * kELT_SIZE), // Clear Size for broadcast stage
            0, 0});
        // If not wait for results, we will rely on the following kernel to update the buffer
    }
}

// This kernel works performant when loads_per_thread is 1.
// For this mode, we are able to support up to 1024 (threads) x 8 (elements) = 8192 hidden dimension.
// There are two options for further scaling up:
//      1. Use CGA if supported. It expands the hidden dimension to 8k x 8 = 64k.
//      2. Set loads_per_thread >1. Which can be used if CGA is not supported. Note that this will be limited by the
//      shared memory size and register count.
template <typename T, ar_fusion::AllReduceFusionPattern Pattern, bool UseCGA, int LoadsPerThread = 1>
__global__ __launch_bounds__(1024) void rmsNormLamport(MnnvlAllReduceKernelParams<T> params)
{
    static int const kELTS_PER_LOAD = sizeof(float4) / sizeof(T);

    uint32_t const token = blockIdx.x;
    uint32_t const blockSize = blockDim.x;
    uint32_t const threadOffset = threadIdx.x;

    uint32_t numThreads = blockSize;
    uint32_t clusterSize = 1;
    uint32_t blockOffset = 0;
    if constexpr (UseCGA)
    {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        namespace cg = cooperative_groups;
        cg::cluster_group cluster = cg::this_cluster();
        numThreads = cluster.num_threads();
        clusterSize = cluster.num_blocks();
        blockOffset = cluster.block_rank();
#endif
    }
    uint32_t const dimPadded
        = divUp(static_cast<uint32_t>(params.tokenDim), kELTS_PER_LOAD * numThreads) * kELTS_PER_LOAD * numThreads;
    uint32_t const elemsPerThread = dimPadded / numThreads;
    uint32_t const loadStride = blockSize;

    extern __shared__ uint8_t smem[];
    float rInput[LoadsPerThread * kELTS_PER_LOAD];
    uint32_t offsets[LoadsPerThread * kELTS_PER_LOAD];

    uint32_t const smemBufferSize = blockSize * elemsPerThread * sizeof(T);
    T* smemInput = reinterpret_cast<T*>(&smem[0]);
    T* smemResidual = reinterpret_cast<T*>(&smem[smemBufferSize]);
    T* smemGamma = reinterpret_cast<T*>(&smem[2 * smemBufferSize]);

    LamportFlags<float4, UseCGA> flag(params.bufferFlags, MNNVLTwoShotStage::NUM_STAGES);
    T* input = reinterpret_cast<T*>(
        flag.getCurLamportBuf(reinterpret_cast<void*>(params.bufferInputPtr), MNNVLTwoShotStage::BROADCAST));

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    // The offset that current thread should load from. Note that the hidden dimension is split by CGA size and each
    // block loads a contiguous chunk;
    // The size of chunk that each block processes
    uint32_t const blockChunkSize
        = divUp(static_cast<uint32_t>(params.tokenDim), clusterSize * kELTS_PER_LOAD) * kELTS_PER_LOAD;
    uint32_t const blockLoadOffset = token * params.tokenDim + blockOffset * blockChunkSize;

#pragma unroll
    for (uint32_t i = 0; i < LoadsPerThread; i++)
    {
        // Each block load a contiguous chunk of tokens
        uint32_t const threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
        offsets[i] = blockLoadOffset + threadLoadOffset;
    }

    if constexpr (ar_fusion::HasResidual<Pattern>)
    {
#pragma unroll
        for (uint32_t i = 0; i < LoadsPerThread; i++)
        {
            uint32_t const threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
            if (blockOffset * blockChunkSize + threadLoadOffset < static_cast<uint32_t>(params.tokenDim))
            {
                copyF4(&smemResidual[threadLoadOffset], &params.residualInPtr[blockLoadOffset + threadLoadOffset]);
            }
        }
        __pipeline_commit();
    }
#pragma unroll
    for (uint32_t i = 0; i < LoadsPerThread; i++)
    {
        uint32_t const threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
        if (blockOffset * blockChunkSize + threadLoadOffset < static_cast<uint32_t>(params.tokenDim))
        {
            copyF4(&smemGamma[threadLoadOffset], &params.gammaPtr[blockOffset * blockChunkSize + threadLoadOffset]);
        }
    }
    __pipeline_commit();

    flag.ctaArrive();
    bool valid = false;
    // ACQBLK if not lamport
    while (!valid)
    {
        valid = true;
#pragma unroll
        for (uint32_t i = 0; i < LoadsPerThread; i++)
        {
            uint32_t threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;

            if (blockOffset * blockChunkSize + threadLoadOffset < static_cast<uint32_t>(params.tokenDim))
            {

                float4* dst4 = reinterpret_cast<float4*>(&smemInput[threadLoadOffset]);
                float4 const* src4 = reinterpret_cast<float4 const*>(&input[offsets[i]]);

                auto loaded = loadPackedVolatile<float4>(src4);
                // Assume that the 16B were written atomically, so we only need to check one value
                valid &= !isLamportDirty(loaded);
                *dst4 = loaded.packed;
            }
        }
    }

    __pipeline_wait_prior(1);
    __syncthreads();

    float threadSum = 0.f;
#pragma unroll
    for (int i = 0; i < LoadsPerThread; i++)
    {
        int threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
        if (blockOffset * blockChunkSize + threadLoadOffset < static_cast<uint32_t>(params.tokenDim))
        {
            PackedVec<float4, T> inp{.packed = loadPacked<float4>(&smemInput[threadLoadOffset])};
            PackedVec<float4, T> inpPlusRes = inp;
            if constexpr (ar_fusion::HasResidual<Pattern>)
            {
                PackedVec<float4, T> res{.packed = loadPacked<float4>(&smemResidual[threadLoadOffset])};
                inpPlusRes = inp + res;
                if constexpr (ar_fusion::HasResidualOut<Pattern>)
                {
                    *reinterpret_cast<float4*>(&params.residualOutPtr[blockLoadOffset + threadLoadOffset])
                        = inpPlusRes.packed;
                }
            }

#pragma unroll
            for (int j = 0; j < kELTS_PER_LOAD; j++)
            {
                rInput[i * kELTS_PER_LOAD + j] = cuda_cast<float, T>(inpPlusRes.elements[j]);
                threadSum += cuda_cast<float, T>(inpPlusRes.elements[j] * inpPlusRes.elements[j]);
            }
        }
    }

    __pipeline_wait_prior(0);

    float blockSum = blockReduceSum<float, true>(threadSum);

    float fullSum = blockSum;
    // Use CGA Reduction if supported
    if constexpr (UseCGA)
    {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        __shared__ float sharedVal[8];
        namespace cg = cooperative_groups;
        cg::cluster_group cluster = cg::this_cluster();
        int const numBlocks = cluster.num_blocks();
        if (numBlocks > 1)
        {
            fullSum = 0.F;
            // Need to reduce over the entire cluster
            int const blockRank = cluster.block_rank();
            if (threadIdx.x < numBlocks)
            {
                cluster.map_shared_rank(&sharedVal[0], threadIdx.x)[blockRank] = blockSum;
            }
            // cluster.sync();
            cluster.barrier_wait(cluster.barrier_arrive());
            for (int i = 0; i < numBlocks; ++i)
            {
                fullSum += sharedVal[i];
            }
        }
#endif
    }

    float rcpRms = rsqrtf(fullSum / params.tokenDim + params.epsilon);

#pragma unroll
    for (int i = 0; i < LoadsPerThread; i++)
    {
        PackedVec<float4, T> r_out;
        uint32_t threadLoadOffset = (i * loadStride + threadOffset) * kELTS_PER_LOAD;
        if (blockOffset * blockChunkSize + threadLoadOffset < static_cast<uint32_t>(params.tokenDim))
        {
            PackedVec<float4, T> gamma = {.packed = loadPacked<float4>(&smemGamma[threadLoadOffset])};

#pragma unroll
            for (uint32_t j = 0; j < kELTS_PER_LOAD; j++)
            {
                r_out.elements[j] = cuda_cast<T, float>(
                    cuda_cast<float, T>(gamma.elements[j]) * rInput[i * kELTS_PER_LOAD + j] * rcpRms);
            }

            int const accessIdInToken = (blockOffset * blockChunkSize + threadLoadOffset) / kELTS_PER_LOAD;
            int const packedAccessIdx = (blockLoadOffset + threadLoadOffset) / kELTS_PER_LOAD;
            writeEpilogueOutput<Pattern, T, float4, kELTS_PER_LOAD>(
                r_out, params, blockLoadOffset + threadLoadOffset, packedAccessIdx, accessIdInToken, token);
        }
    }
    constexpr int kELTS_SIZE = sizeof(T);

    // Issue ACQBLK at the end. Assuming preceding kernel will not modify the buffer_flags.
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    // Update the buffer pointers
    flag.waitAndUpdate({static_cast<uint32_t>(divUp<uint32_t>(params.numTokens, params.nRanks) * params.nRanks
                            * params.tokenDim * kELTS_SIZE),
        static_cast<uint32_t>(params.numTokens * params.tokenDim * kELTS_SIZE), 0, 0});
}

void twoshotAllreduceFusionOp(AllReduceFusionParams const& params)
{
    static int const kSMVersion = tensorrt_llm::common::getSMVersion();
    TLLM_CHECK_WITH_INFO(kSMVersion >= 90, "[MNNVL AllReduceTwoShot] requires SM 90 or newer.");
    int const numTokens = params.numTokens;
    int const tokenDim = params.tokenDim;
    int const numEltsPerThread = sizeof(float4) / getDTypeSize(params.dType);
    TLLM_CHECK_WITH_INFO(tokenDim % numEltsPerThread == 0, "[MNNVL AllReduceTwoShot] token_dim must be divisible by %d",
        numEltsPerThread);

    int const arNumThreads = divUp(tokenDim, numEltsPerThread);
    int const arNumBlocksPerToken = divUp(arNumThreads, 128);

    dim3 arGrid(numTokens, arNumBlocksPerToken);

    cudaLaunchAttribute arAttrs[1];
    arAttrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    arAttrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;

    cudaLaunchConfig_t arConfig{
        .gridDim = arGrid,
        .blockDim = 128,
        .dynamicSmemBytes = 0,
        .stream = params.stream,
        .attrs = arAttrs,
        .numAttrs = 1,
    };

    TLLM_LOG_DEBUG(
        "[MNNVL AllReduceTwoShot] Dispatch: grid size: (%d, %d, 1), block_size: 128", numTokens, arNumBlocksPerToken);

#define LAUNCH_ALLREDUCE_KERNEL(WORLD_SIZE, T)                                                                         \
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&arConfig, &twoshotAllreduceKernel<WORLD_SIZE, T>, kernelParams));
    auto dispatchAR = [&](auto* type_ptr) -> bool
    {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        T** ucPtrs = reinterpret_cast<T**>(params.bufferPtrsDev);
        T* mcastPtr = reinterpret_cast<T*>(params.multicastPtr);
        T* output = reinterpret_cast<T*>(params.output);
        T const* input = reinterpret_cast<T const*>(params.input);
        MnnvlAllReduceKernelParams<T> kernelParams{output, reinterpret_cast<T*>(params.residualOut), input,
            reinterpret_cast<T const*>(params.residualIn), reinterpret_cast<T const*>(params.gamma), ucPtrs,
            reinterpret_cast<T*>(params.bufferPtrLocal), mcastPtr, params.quantOut, params.scaleOut, params.scaleFactor,
            numTokens, tokenDim, params.nRanks, params.rank, static_cast<float>(params.epsilon), params.bufferFlags,
            !params.rmsNormFusion, params.layout};
        switch (params.nRanks)
        {
        case 2: LAUNCH_ALLREDUCE_KERNEL(2, T); return true;
        case 4: LAUNCH_ALLREDUCE_KERNEL(4, T); return true;
        case 8: LAUNCH_ALLREDUCE_KERNEL(8, T); return true;
        case 16: LAUNCH_ALLREDUCE_KERNEL(16, T); return true;
        case 32: LAUNCH_ALLREDUCE_KERNEL(32, T); return true;
        case 64: LAUNCH_ALLREDUCE_KERNEL(64, T); return true;
        }
        return false;
    };

#undef LAUNCH_ALLREDUCE_KERNEL

    bool launched = (params.dType == tensorrt_llm::DataType::kFLOAT && dispatchAR((float*) nullptr))
        || (params.dType == tensorrt_llm::DataType::kBF16 && dispatchAR((__nv_bfloat16*) nullptr))
        || (params.dType == tensorrt_llm::DataType::kHALF && dispatchAR((__nv_half*) nullptr));
    if (!launched)
    {
        TLLM_CHECK_WITH_INFO(false, "[MNNVL AllReduceTwoShot] Failed to dispatch twoshotAllreduce kernel.");
    }
    // Launch the rmsnorm lamport kernel if fusion is enabled
    if (params.rmsNormFusion)
    {
        int const accessGroupSize = requiresTwoAccessScaleGroup(params.pattern) ? 2 : 1;
        auto gridConfig = adjustGridConfig<true>(numTokens, tokenDim, numEltsPerThread, accessGroupSize);
        int rnBlockSize = std::get<0>(gridConfig);
        int rnClusterSize = std::get<1>(gridConfig);
        int rnLoadsPerThread = std::get<2>(gridConfig);
        bool rnUseCGA = rnClusterSize > 1 && rnLoadsPerThread == 1;
        if (!rnUseCGA)
        {
            gridConfig = adjustGridConfig<false>(numTokens, tokenDim, numEltsPerThread, accessGroupSize);
            rnBlockSize = std::get<0>(gridConfig);
            rnClusterSize = std::get<1>(gridConfig);
            rnLoadsPerThread = std::get<2>(gridConfig);
        }

        int rnNumThreads = rnClusterSize * rnBlockSize;
        dim3 rnGrid(numTokens, rnClusterSize, 1);
        cudaLaunchConfig_t rnConfig;
        cudaLaunchAttribute rnAttrs[2];
        rnConfig.stream = params.stream;
        rnConfig.gridDim = rnGrid;
        rnConfig.blockDim = rnBlockSize;
        rnConfig.attrs = rnAttrs;
        rnAttrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        rnAttrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
        rnConfig.numAttrs = 1U;
        if (rnUseCGA)
        {
            rnAttrs[1].id = cudaLaunchAttributeClusterDimension;
            rnAttrs[1].val.clusterDim.x = 1;
            rnAttrs[1].val.clusterDim.y = rnClusterSize;
            rnAttrs[1].val.clusterDim.z = 1;
            rnConfig.numAttrs = 2U;
        }
        int const dimPadded = divUp(tokenDim, numEltsPerThread * rnNumThreads) * numEltsPerThread * rnNumThreads;
        int const iters = dimPadded / rnNumThreads;

        size_t const smemSize = 3 * rnBlockSize * iters * getDTypeSize(params.dType);

        TLLM_LOG_DEBUG(
            "[MNNVL AllReduceTwoShotRMSNorm] Dispatch: grid size: (%d, %d, 1), block_size: %d, cluster_size: %d, "
            "loads_per_thread: %d, "
            "threads_needed: %d",
            numTokens, rnClusterSize, rnBlockSize, rnClusterSize, rnLoadsPerThread, divUp(tokenDim, numEltsPerThread));

#define RUN_RMSNORM_KERNEL(T, PATTERN, USE_CGA, LOADS_PER_THREAD)                                                      \
    TLLM_CUDA_CHECK(cudaFuncSetAttribute(&rmsNormLamport<T, PATTERN, USE_CGA, LOADS_PER_THREAD>,                       \
        cudaFuncAttributeMaxDynamicSharedMemorySize, smemSize));                                                       \
    rnConfig.dynamicSmemBytes = smemSize;                                                                              \
    TLLM_CUDA_CHECK(                                                                                                   \
        cudaLaunchKernelEx(&rnConfig, &rmsNormLamport<T, PATTERN, USE_CGA, LOADS_PER_THREAD>, kernelParams));

#define DISPATCH_RMSNORM_PATTERN(T, USE_CGA, LOADS_PER_THREAD)                                                         \
    if (params.pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNorm)                                       \
    {                                                                                                                  \
        RUN_RMSNORM_KERNEL(T, ar_fusion::AllReduceFusionPattern::kARResidualRMSNorm, USE_CGA, LOADS_PER_THREAD);       \
    }                                                                                                                  \
    else if (params.pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNormFP8Quant)                          \
    {                                                                                                                  \
        RUN_RMSNORM_KERNEL(                                                                                            \
            T, ar_fusion::AllReduceFusionPattern::kARResidualRMSNormFP8Quant, USE_CGA, LOADS_PER_THREAD);              \
    }                                                                                                                  \
    else if (params.pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP8Quant)                       \
    {                                                                                                                  \
        RUN_RMSNORM_KERNEL(                                                                                            \
            T, ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP8Quant, USE_CGA, LOADS_PER_THREAD);           \
    }                                                                                                                  \
    else if (params.pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNormFP4Quant)                          \
    {                                                                                                                  \
        if constexpr (!std::is_same_v<T, float>)                                                                       \
        {                                                                                                              \
            RUN_RMSNORM_KERNEL(                                                                                        \
                T, ar_fusion::AllReduceFusionPattern::kARResidualRMSNormFP4Quant, USE_CGA, LOADS_PER_THREAD);          \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            TLLM_CHECK_WITH_INFO(false, "[MNNVL AllReduceTwoShot] NVFP4 quantization does not support FP32 input.");   \
        }                                                                                                              \
    }                                                                                                                  \
    else if (params.pattern == ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP4Quant)                       \
    {                                                                                                                  \
        if constexpr (!std::is_same_v<T, float>)                                                                       \
        {                                                                                                              \
            RUN_RMSNORM_KERNEL(                                                                                        \
                T, ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP4Quant, USE_CGA, LOADS_PER_THREAD);       \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            TLLM_CHECK_WITH_INFO(false, "[MNNVL AllReduceTwoShot] NVFP4 quantization does not support FP32 input.");   \
        }                                                                                                              \
    }                                                                                                                  \
    else if (params.pattern == ar_fusion::AllReduceFusionPattern::kARRMSNorm)                                          \
    {                                                                                                                  \
        RUN_RMSNORM_KERNEL(T, ar_fusion::AllReduceFusionPattern::kARRMSNorm, USE_CGA, LOADS_PER_THREAD);               \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        TLLM_CHECK_WITH_INFO(false, "[MNNVL AllReduceTwoShot] Unsupported RMSNorm fusion pattern.");                   \
    }

        // C++ 17 does not support capturing structured bindings
        auto dispatchRN = [&, rnLoadsPerThread](auto* type_ptr)
        {
            using T = std::remove_pointer_t<decltype(type_ptr)>;
            MnnvlAllReduceKernelParams<T> kernelParams{reinterpret_cast<T*>(params.output),
                reinterpret_cast<T*>(params.residualOut), reinterpret_cast<T const*>(params.input),
                reinterpret_cast<T const*>(params.residualIn), reinterpret_cast<T const*>(params.gamma),
                reinterpret_cast<T**>(params.bufferPtrsDev), reinterpret_cast<T*>(params.bufferPtrLocal),
                reinterpret_cast<T*>(params.multicastPtr), params.quantOut, params.scaleOut, params.scaleFactor,
                numTokens, tokenDim, params.nRanks, params.rank, static_cast<float>(params.epsilon), params.bufferFlags,
                false, params.layout};
            if (rnUseCGA)
            {
                DISPATCH_RMSNORM_PATTERN(T, true, 1);
            }
            else
            {
                switch (rnLoadsPerThread)
                {
                case 1: DISPATCH_RMSNORM_PATTERN(T, false, 1); break;
                case 2: DISPATCH_RMSNORM_PATTERN(T, false, 2); break;
                case 3: DISPATCH_RMSNORM_PATTERN(T, false, 3); break;
                case 4: DISPATCH_RMSNORM_PATTERN(T, false, 4); break;
                case 5: DISPATCH_RMSNORM_PATTERN(T, false, 5); break;
                case 6: DISPATCH_RMSNORM_PATTERN(T, false, 6); break;
                case 7: DISPATCH_RMSNORM_PATTERN(T, false, 7); break;
                case 8: DISPATCH_RMSNORM_PATTERN(T, false, 8); break;
                default: return false;
                }
            }
            return true;
        };

        launched = (params.dType == tensorrt_llm::DataType::kFLOAT && dispatchRN((float*) nullptr))
            || (params.dType == tensorrt_llm::DataType::kBF16 && dispatchRN((__nv_bfloat16*) nullptr))
            || (params.dType == tensorrt_llm::DataType::kHALF && dispatchRN((__nv_half*) nullptr));
        if (!launched)
        {
            TLLM_CHECK_WITH_INFO(false, "[MNNVL AllReduceTwoShot] Failed to dispatch rmsnorm lamport kernel.");
        }
#undef RUN_RMSNORM_KERNEL
#undef DISPATCH_RMSNORM_PATTERN
    }
}

// ============================== All-reduce + wide residual ==============================
//
// The all-reduce's sum is the update of a wide residual step: it is written into the residual streams and never
// stored on its own. One CTA per token row, one thread per eight columns, for every row count and on both the
// one-shot and the two-shot path, so the variance of a normed read is always summed in the same order and a row's
// outputs do not depend on how many rows share the call.

//! The most threads a wide-residual CTA has: one per eight columns, so a hidden width of at most 3072.
constexpr int kMaxWideResidualThreads = 384;

namespace detail
{

struct WideResidualKernelParams
{
    __nv_bfloat16 const* shardPtr;
    __nv_bfloat16** inputPtrs;
    __nv_bfloat16* bufferInputPtr;
    __nv_bfloat16* mcastPtr;
    uint32_t* bufferFlags;
    __nv_bfloat16* streams;
    int64_t streamsRowStride;
    __nv_bfloat16 const* retention;
    __nv_bfloat16 const* write;
    __nv_bfloat16 const* readFactors;
    __nv_bfloat16 const* normWeight;
    __nv_bfloat16 const* readFactors2;
    __nv_bfloat16 const* normWeight2;
    __nv_bfloat16* normedOut;
    __nv_bfloat16* normedOut2;
    float epsilon;
    float invTokenDim;
    int numTokens;
    int tokenDim;
    int nRanks;
    int rank;
};

//! Sum `kN` values over the CTA in an order set by the CTA's shape alone: within each warp by butterfly, then the
//! warps' partials in warp order. Every thread receives the sums. blockDim.x must be a multiple of 32 and at most
//! kMaxWideResidualThreads; one call per CTA.
template <int kN>
inline __device__ void blockReduceSumFixedOrder(float (&v)[kN])
{
    constexpr int kMaxWarps = kMaxWideResidualThreads / static_cast<int>(kWARP_SIZE);
    __shared__ float partials[kN][kMaxWarps];
    int const lane = threadIdx.x & kLANE_ID_MASK;
    int const warp = threadIdx.x >> kLOG2_WARP_SIZE;
    int const numWarps = blockDim.x >> kLOG2_WARP_SIZE;
#pragma unroll
    for (int i = 0; i < kN; i++)
    {
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
        {
            v[i] += __shfl_xor_sync(0xffffffffU, v[i], mask, kWARP_SIZE);
        }
        if (lane == 0)
        {
            partials[i][warp] = v[i];
        }
    }
    __syncthreads();
    // All partials are loaded before the first add; a missing warp contributes +0, which leaves a sum of squares as
    // it is.
    float loaded[kN][kMaxWarps];
#pragma unroll
    for (int w = 0; w < kMaxWarps; w++)
    {
#pragma unroll
        for (int i = 0; i < kN; i++)
        {
            loaded[i][w] = w < numWarps ? partials[i][w] : 0.F;
        }
    }
#pragma unroll
    for (int i = 0; i < kN; i++)
    {
        float sum = 0.F;
#pragma unroll
        for (int w = 0; w < kMaxWarps; w++)
        {
            sum += loaded[i][w];
        }
        v[i] = sum;
    }
}

//! The step's parameters for this thread's columns: the norm weights, and the coefficients as fp32 copies of their
//! bf16 values. Nothing in a forward writes them, so they may be read before the kernel ahead has finished.
template <int kStreams, bool TwoReads>
inline __device__ void loadWideResidualParameters(WideResidualKernelParams const& params, int col, bool inBounds,
    PackedVec<float4, __nv_bfloat16>& g0, PackedVec<float4, __nv_bfloat16>& g1, float (&r)[kStreams],
    float (&w)[kStreams], float (&f0)[kStreams], float (&f1)[kStreams])
{
    if (inBounds)
    {
        g0.packed = loadPacked<float4>(&params.normWeight[col]);
        if constexpr (TwoReads)
        {
            g1.packed = loadPacked<float4>(&params.normWeight2[col]);
        }
    }
#pragma unroll
    for (int k = 0; k < kStreams; k++)
    {
        r[k] = __bfloat162float(params.retention[k]);
        w[k] = __bfloat162float(params.write[k]);
        f0[k] = __bfloat162float(params.readFactors[k]);
        f1[k] = TwoReads ? __bfloat162float(params.readFactors2[k]) : 0.F;
    }
}

//! This thread's eight columns of the token's stream rows.
template <int kStreams>
inline __device__ void loadStreamRows(WideResidualKernelParams const& params, int token, int col, bool inBounds,
    PackedVec<float4, __nv_bfloat16> (&s)[kStreams])
{
    if (inBounds)
    {
        __nv_bfloat16 const* row = params.streams + token * params.streamsRowStride + col;
#pragma unroll
        for (int k = 0; k < kStreams; k++)
        {
            s[k].packed = __ldcg(reinterpret_cast<float4 const*>(row + static_cast<int64_t>(k) * params.tokenDim));
        }
    }
}

//! The wide residual step on this thread's eight columns, given the all-reduced update `u`: each stream written,
//! `bf16(bf16(r * s) + u * w)`, in registers and in place; each read `x = bf16(sum_k f_k * s_k)` over the written
//! streams in stream order; then `bf16(g * bf16(x * rsqrt(sum(x^2) / tokenDim + eps)))`. Every product of two bf16
//! values is exact in fp32, so a fused multiply-add rounds as the separate multiply and add do. Every thread of the
//! CTA must call this (it reduces across the CTA); a thread past the row's end stores nothing.
template <int kStreams, bool TwoReads>
inline __device__ void wideResidualEpilogue(WideResidualKernelParams const& params, int token, int col, bool inBounds,
    PackedVec<float4, __nv_bfloat16> const& u, PackedVec<float4, __nv_bfloat16> const (&s)[kStreams],
    float const (&r)[kStreams], float const (&w)[kStreams], float const (&f0)[kStreams], float const (&f1)[kStreams],
    PackedVec<float4, __nv_bfloat16> const& g0, PackedVec<float4, __nv_bfloat16> const& g1)
{
    constexpr int kELTS = 8;
    constexpr int kReads = TwoReads ? 2 : 1;
    float x[kReads][kELTS];
    float squares[kReads];
#pragma unroll
    for (int j = 0; j < kReads; j++)
    {
        squares[j] = 0.F;
    }
    if (inBounds)
    {
        __nv_bfloat16* row = params.streams + token * params.streamsRowStride + col;
        float acc[kReads][kELTS];
#pragma unroll
        for (int k = 0; k < kStreams; k++)
        {
            PackedVec<float4, __nv_bfloat16> written;
#pragma unroll
            for (int i = 0; i < kELTS; i++)
            {
                float const kept = __bfloat162float(__float2bfloat16_rn(__bfloat162float(s[k].elements[i]) * r[k]));
                written.elements[i] = __float2bfloat16_rn(kept + __bfloat162float(u.elements[i]) * w[k]);
                float const sk = __bfloat162float(written.elements[i]);
                acc[0][i] = k == 0 ? f0[k] * sk : acc[0][i] + f0[k] * sk;
                if constexpr (TwoReads)
                {
                    acc[1][i] = k == 0 ? f1[k] * sk : acc[1][i] + f1[k] * sk;
                }
            }
            *reinterpret_cast<float4*>(row + static_cast<int64_t>(k) * params.tokenDim) = written.packed;
        }
#pragma unroll
        for (int j = 0; j < kReads; j++)
        {
#pragma unroll
            for (int i = 0; i < kELTS; i++)
            {
                x[j][i] = __bfloat162float(__float2bfloat16_rn(acc[j][i]));
                squares[j] += x[j][i] * x[j][i];
            }
        }
    }
    blockReduceSumFixedOrder<kReads>(squares);
    if (inBounds)
    {
        int64_t const offset = static_cast<int64_t>(token) * params.tokenDim + col;
#pragma unroll
        for (int j = 0; j < kReads; j++)
        {
            PackedVec<float4, __nv_bfloat16> const& g = j == 0 ? g0 : g1;
            float const rcpRms = rsqrtf(squares[j] * params.invTokenDim + params.epsilon);
            PackedVec<float4, __nv_bfloat16> y;
#pragma unroll
            for (int i = 0; i < kELTS; i++)
            {
                float const normed = __bfloat162float(__float2bfloat16_rn(x[j][i] * rcpRms));
                y.elements[i] = __float2bfloat16_rn(__bfloat162float(g.elements[i]) * normed);
            }
            *reinterpret_cast<float4*>((j == 0 ? params.normedOut : params.normedOut2) + offset) = y.packed;
        }
    }
}

} // namespace detail

using detail::WideResidualKernelParams;
using detail::loadWideResidualParameters;
using detail::loadStreamRows;
using detail::wideResidualEpilogue;

template <uint8_t WorldSize, int kStreams, bool TwoReads>
__global__ void __launch_bounds__(kMaxWideResidualThreads)
    oneshotAllreduceWideResidualKernel(WideResidualKernelParams params)
{
    using T = __nv_bfloat16;
    using PackedType = float4;
    constexpr int kELTS_PER_THREAD = sizeof(PackedType) / sizeof(T);
    constexpr uint32_t kELT_SIZE = sizeof(T);
    int const token = blockIdx.x;
    int const packedIdx = threadIdx.x;
    int const col = packedIdx * kELTS_PER_THREAD;
    bool const inBounds = col < params.tokenDim;

    PackedVec<PackedType, T> g0, g1;
    float r[kStreams], w[kStreams], f0[kStreams], f1[kStreams];
    loadWideResidualParameters<kStreams, TwoReads>(params, col, inBounds, g0, g1, r, w, f0, f1);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
    // The next kernel may launch now: it waits for this one to finish before it reads anything this one writes, and
    // its launch overlaps the wait for the other ranks below.
    cudaTriggerProgrammaticLaunchCompletion();
#endif

    // We only use 1 stage for the oneshot allreduce
    LamportFlags<PackedType, false> flag(params.bufferFlags, 1);
    T* stagePtrMcast = reinterpret_cast<T*>(flag.getCurLamportBuf(params.mcastPtr, 0));
    T* stagePtrLocal = reinterpret_cast<T*>(flag.getCurLamportBuf(params.inputPtrs[params.rank], 0));

    // ==================== Broadcast tokens to each rank =============================
    PackedVec<PackedType, T> val;
    if (inBounds)
    {
        val.packed = loadPacked<PackedType>(&params.shardPtr[static_cast<int64_t>(token) * params.tokenDim + col]);
        sanitizeLamportPayload(val);
        reinterpret_cast<PackedType*>(
            &stagePtrMcast[token * params.tokenDim * WorldSize + params.rank * params.tokenDim])[packedIdx]
            = val.packed;
    }
    // The row's streams, loaded while the other ranks' partials are in flight.
    PackedVec<PackedType, T> s[kStreams];
    loadStreamRows<kStreams>(params, token, col, inBounds, s);

    flag.ctaArrive();
    // ======================= Lamport Sync and clear the output buffer from previous iteration
    flag.clearDirtyLamportBuf(params.inputPtrs[params.rank], -1);

    // ======================= Reduction: allreduce's sum, bit for bit =============================
    PackedVec<PackedType, T> u;
    if (inBounds)
    {
        u = val;
#define RUN_ONESHOT_LOCAL_RANK(LOCAL_RANK)                                                                             \
    case LOCAL_RANK:                                                                                                   \
        if constexpr (WorldSize > LOCAL_RANK)                                                                          \
        {                                                                                                              \
            u = reduceOneshotDeterministicFastPath<WorldSize, LOCAL_RANK, T, PackedType, kELTS_PER_THREAD>(            \
                val, stagePtrLocal, token, params.tokenDim, packedIdx);                                                \
        }                                                                                                              \
        break

        switch (params.rank)
        {
            RUN_ONESHOT_LOCAL_RANK(0);
            RUN_ONESHOT_LOCAL_RANK(1);
            RUN_ONESHOT_LOCAL_RANK(2);
            RUN_ONESHOT_LOCAL_RANK(3);
            RUN_ONESHOT_LOCAL_RANK(4);
            RUN_ONESHOT_LOCAL_RANK(5);
            RUN_ONESHOT_LOCAL_RANK(6);
            RUN_ONESHOT_LOCAL_RANK(7);
        }
#undef RUN_ONESHOT_LOCAL_RANK
    }
    // =============================== Wide residual ===============================
    wideResidualEpilogue<kStreams, TwoReads>(params, token, col, inBounds, u, s, r, w, f0, f1, g0, g1);
    flag.waitAndUpdate({static_cast<uint32_t>(params.numTokens * params.tokenDim * WorldSize * kELT_SIZE), 0, 0, 0});
}

//! The two-shot path's second kernel, in place of rmsNormLamport: waits for each element of the row's broadcast
//! sum, then runs the wide residual step on it. Launched right behind twoshotAllreduceKernel with
//! waitForResults = false. That kernel triggers only after its own dependency wait, so every kernel before the
//! all-reduce has finished when this one starts, and the streams (which the all-reduce does not touch) can be read
//! before the sums arrive.
//!
//! One CTA per row. At prefill heights the kernel is a stream of short CTAs, so its throughput is how many fit on an
//! SM at once: the bound keeps a CTA of the widest row this op takes (3072 columns) at three per SM.
template <int kStreams, bool TwoReads>
__global__ __launch_bounds__(kMaxWideResidualThreads, 3) void wideResidualLamport(WideResidualKernelParams params)
{
    using T = __nv_bfloat16;
    constexpr int kELTS_PER_THREAD = sizeof(float4) / sizeof(T);
    constexpr uint32_t kELT_SIZE = sizeof(T);
    int const token = blockIdx.x;
    int const col = threadIdx.x * kELTS_PER_THREAD;
    bool const inBounds = col < params.tokenDim;

    LamportFlags<float4, false> flag(params.bufferFlags, MNNVLTwoShotStage::NUM_STAGES);
    T const* sums = reinterpret_cast<T const*>(
        flag.getCurLamportBuf(reinterpret_cast<void*>(params.bufferInputPtr), MNNVLTwoShotStage::BROADCAST));
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif

    PackedVec<float4, T> g0, g1;
    float r[kStreams], w[kStreams], f0[kStreams], f1[kStreams];
    loadWideResidualParameters<kStreams, TwoReads>(params, col, inBounds, g0, g1, r, w, f0, f1);
    PackedVec<float4, T> s[kStreams];
    loadStreamRows<kStreams>(params, token, col, inBounds, s);

    flag.ctaArrive();
    PackedVec<float4, T> u;
    if (inBounds)
    {
        auto loaded = loadPackedVolatile<float4>(&sums[static_cast<int64_t>(token) * params.tokenDim + col]);
        while (isLamportDirty(loaded))
        {
            loaded = loadPackedVolatile<float4>(&sums[static_cast<int64_t>(token) * params.tokenDim + col]);
        }
        u.packed = loaded.packed;
    }
    wideResidualEpilogue<kStreams, TwoReads>(params, token, col, inBounds, u, s, r, w, f0, f1, g0, g1);

    // As in rmsNormLamport: the kernel before must be done before the buffer flags move on.
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    flag.waitAndUpdate({static_cast<uint32_t>(divUp<uint32_t>(params.numTokens, params.nRanks) * params.nRanks
                            * params.tokenDim * kELT_SIZE),
        static_cast<uint32_t>(params.numTokens * params.tokenDim * kELT_SIZE), 0, 0});
}

namespace
{
constexpr int kWideResidualStreams = 3;

//! One thread per eight columns, rounded up to whole warps: the same CTA for every row count.
int wideResidualBlockSize(int tokenDim)
{
    int const threads = divUp(tokenDim, static_cast<int>(sizeof(float4) / sizeof(__nv_bfloat16)));
    return divUp(threads, static_cast<int>(detail::kWARP_SIZE)) * static_cast<int>(detail::kWARP_SIZE);
}

void checkWideResidualParams(AllReduceWideResidualParams const& params, char const* name)
{
    static int const kSMVersion = tensorrt_llm::common::getSMVersion();
    TLLM_CHECK_WITH_INFO(kSMVersion >= 90, "[%s] requires SM 90 or newer.", name);
    TLLM_CHECK_WITH_INFO(params.numStreams == kWideResidualStreams, "[%s] supports %d residual streams, got %d.", name,
        kWideResidualStreams, params.numStreams);
    TLLM_CHECK_WITH_INFO(
        params.tokenDim % 8 == 0, "[%s] token_dim must be divisible by 8, got %d.", name, params.tokenDim);
    TLLM_CHECK_WITH_INFO(wideResidualBlockSize(params.tokenDim) <= kMaxWideResidualThreads,
        "[%s] token_dim %d exceeds %d.", name, params.tokenDim, kMaxWideResidualThreads * 8);
    TLLM_CHECK_WITH_INFO((params.readFactors2 == nullptr) == (params.normWeight2 == nullptr)
            && (params.readFactors2 == nullptr) == (params.normedOut2 == nullptr),
        "[%s] a second read needs its factors, its norm weight and its output.", name);
}

WideResidualKernelParams toKernelParams(AllReduceWideResidualParams const& params)
{
    using T = __nv_bfloat16;
    return WideResidualKernelParams{reinterpret_cast<T const*>(params.input),
        reinterpret_cast<T**>(params.bufferPtrsDev), reinterpret_cast<T*>(params.bufferPtrLocal),
        reinterpret_cast<T*>(params.multicastPtr), params.bufferFlags, reinterpret_cast<T*>(params.streams),
        params.streamsRowStride, reinterpret_cast<T const*>(params.retention), reinterpret_cast<T const*>(params.write),
        reinterpret_cast<T const*>(params.readFactors), reinterpret_cast<T const*>(params.normWeight),
        reinterpret_cast<T const*>(params.readFactors2), reinterpret_cast<T const*>(params.normWeight2),
        reinterpret_cast<T*>(params.normedOut), reinterpret_cast<T*>(params.normedOut2), params.epsilon,
        params.invTokenDim, params.numTokens, params.tokenDim, params.nRanks, params.rank};
}
} // namespace

void oneshotAllreduceWideResidualOp(AllReduceWideResidualParams const& params)
{
    checkWideResidualParams(params, "MNNVL AllReduceWideResidualOneShot");
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    cudaLaunchConfig_t config{
        .gridDim = dim3(params.numTokens, 1, 1),
        .blockDim = dim3(wideResidualBlockSize(params.tokenDim), 1, 1),
        .dynamicSmemBytes = 0,
        .stream = params.stream,
        .attrs = attrs,
        .numAttrs = 1U,
    };
    WideResidualKernelParams const kernelParams = toKernelParams(params);
    bool const twoReads = params.readFactors2 != nullptr;

#define LAUNCH_WIDE_RESIDUAL_KERNEL(WORLD_SIZE)                                                                        \
    if (twoReads)                                                                                                      \
    {                                                                                                                  \
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(                                                                            \
            &config, &oneshotAllreduceWideResidualKernel<WORLD_SIZE, kWideResidualStreams, true>, kernelParams));      \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(                                                                            \
            &config, &oneshotAllreduceWideResidualKernel<WORLD_SIZE, kWideResidualStreams, false>, kernelParams));     \
    }

    switch (params.nRanks)
    {
    case 2: LAUNCH_WIDE_RESIDUAL_KERNEL(2); break;
    case 4: LAUNCH_WIDE_RESIDUAL_KERNEL(4); break;
    case 8: LAUNCH_WIDE_RESIDUAL_KERNEL(8); break;
    default:
        TLLM_CHECK_WITH_INFO(
            false, "[MNNVL AllReduceWideResidualOneShot] supports 2, 4 or 8 ranks, got %d.", params.nRanks);
    }
#undef LAUNCH_WIDE_RESIDUAL_KERNEL
}

void twoshotAllreduceWideResidualOp(AllReduceWideResidualParams const& params)
{
    using T = __nv_bfloat16;
    checkWideResidualParams(params, "MNNVL AllReduceWideResidualTwoShot");
    int const numTokens = params.numTokens;
    int const tokenDim = params.tokenDim;

    // The two-shot all-reduce, leaving each row's sum in the broadcast stage for the kernel below.
    int const arNumThreads = divUp(tokenDim, static_cast<int>(sizeof(float4) / sizeof(T)));
    cudaLaunchAttribute arAttrs[1];
    arAttrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    arAttrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    cudaLaunchConfig_t arConfig{
        .gridDim = dim3(numTokens, divUp(arNumThreads, 128), 1),
        .blockDim = 128,
        .dynamicSmemBytes = 0,
        .stream = params.stream,
        .attrs = arAttrs,
        .numAttrs = 1U,
    };
    MnnvlAllReduceKernelParams<T> arParams{nullptr, nullptr, reinterpret_cast<T const*>(params.input), nullptr, nullptr,
        reinterpret_cast<T**>(params.bufferPtrsDev), reinterpret_cast<T*>(params.bufferPtrLocal),
        reinterpret_cast<T*>(params.multicastPtr), nullptr, nullptr, nullptr, numTokens, tokenDim, params.nRanks,
        params.rank, params.epsilon, params.bufferFlags, false, QuantizationSFLayout::SWIZZLED};
    switch (params.nRanks)
    {
    case 2: TLLM_CUDA_CHECK(cudaLaunchKernelEx(&arConfig, &twoshotAllreduceKernel<2, T>, arParams)); break;
    case 4: TLLM_CUDA_CHECK(cudaLaunchKernelEx(&arConfig, &twoshotAllreduceKernel<4, T>, arParams)); break;
    case 8: TLLM_CUDA_CHECK(cudaLaunchKernelEx(&arConfig, &twoshotAllreduceKernel<8, T>, arParams)); break;
    default:
        TLLM_CHECK_WITH_INFO(
            false, "[MNNVL AllReduceWideResidualTwoShot] supports 2, 4 or 8 ranks, got %d.", params.nRanks);
    }

    // Then each row's wide residual step, as soon as the row's sum lands.
    cudaLaunchAttribute wrAttrs[1];
    wrAttrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    wrAttrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    cudaLaunchConfig_t wrConfig{
        .gridDim = dim3(numTokens, 1, 1),
        .blockDim = dim3(wideResidualBlockSize(tokenDim), 1, 1),
        .dynamicSmemBytes = 0,
        .stream = params.stream,
        .attrs = wrAttrs,
        .numAttrs = 1U,
    };
    WideResidualKernelParams const kernelParams = toKernelParams(params);
    if (params.readFactors2 != nullptr)
    {
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(&wrConfig, &wideResidualLamport<kWideResidualStreams, true>, kernelParams));
    }
    else
    {
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(&wrConfig, &wideResidualLamport<kWideResidualStreams, false>, kernelParams));
    }
}

} // namespace kernels::mnnvl

TRTLLM_NAMESPACE_END
