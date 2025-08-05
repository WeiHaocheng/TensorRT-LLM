/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/fusedMoeCommKernels.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"

namespace tensorrt_llm
{
namespace kernels
{

static __device__ __forceinline__ uint32_t __as_ptr_smem(void const* __ptr)
{
    // Consider adding debug asserts here.
    return static_cast<uint32_t>(__cvta_generic_to_shared(__ptr));
}

static __device__ __forceinline__ uint64_t __as_ptr_gmem(void const* __ptr)
{
    // Consider adding debug asserts here.
    return static_cast<uint64_t>(__cvta_generic_to_global(__ptr));
}

__device__ __forceinline__ void mbarrier_init(uint64_t* addr, uint32_t const& count)
{
#if defined(__CUDACC__) || __CUDA_ARCH__ >= 800
    asm("mbarrier.init.shared.b64 [%0], %1;" : : "r"(__as_ptr_smem(addr)), "r"(count) : "memory");
#endif
}

__device__ __forceinline__ void mbarrier_expect_tx(uint64_t* addr, const uint32_t txCount)
{
#if defined(__CUDACC__) || __CUDA_ARCH__ >= 900
    asm("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
        :
        : "r"(__as_ptr_smem(addr)), "r"(txCount)
        : "memory");
#endif
}

__device__ __forceinline__ uint64_t mbarrier_arrive(uint64_t* addr)
{
#if defined(__CUDACC__) || __CUDA_ARCH__ >= 800
    uint64_t state;
    asm("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(state) : "r"(__as_ptr_smem(addr)) : "memory");
    return state;
#else
    return 0;
#endif
}

__device__ __forceinline__ uint64_t mbarrier_arrive_expect_tx(uint64_t* addr, const uint32_t txCount)
{
#if defined(__CUDACC__) || __CUDA_ARCH__ >= 900
    uint64_t state;
    asm("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;"
        : "=l"(state)
        : "r"(__as_ptr_smem(addr)), "r"(txCount)
        : "memory");
    return state;
#else
    return 0;
#endif
}

__device__ __forceinline__ bool mbarrier_try_wait_parity(uint64_t* addr, uint32_t const& phaseParity)
{
#if defined(__CUDACC__) || __CUDA_ARCH__ >= 900
    uint32_t waitComplete;
    asm("{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64  P_OUT, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(waitComplete)
        : "r"(__as_ptr_smem(addr)), "r"(phaseParity)
        : "memory");
    return static_cast<bool>(waitComplete);
#else
    return false;
#endif
}

template <int COPY_SIZE = 4>
__device__ __forceinline__ void ldgsts(int* dstShm, int const* srcMem, bool predGuard)
{
#if defined(__CUDACC__) || __CUDA_ARCH__ >= 800
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %0, 0;\n"
        "  @p cp.async.ca.shared.global [%1], [%2], %3;\n"
        "}\n" ::"r"((int) predGuard),
        "r"(__as_ptr_smem(dstShm)), "l"(__as_ptr_gmem(srcMem)), "n"(COPY_SIZE));
#endif
}

__device__ __forceinline__ void cp_async_commit_group()
{
#if defined(__CUDACC__) || __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" : : :);
#endif
}

template <int N = 0>
__device__ __forceinline__ void cp_async_wait_group()
{
#if defined(__CUDACC__) || __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;" : : "n"(N) : "memory");
#endif
}

__device__ __forceinline__ void cp_async_bulk_g2s(void* dstMem, void const* srcMem, int copySize, uint64_t* smemBar)
{
#if defined(__CUDACC__) || __CUDA_ARCH__ >= 900
    asm("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        :
        : "r"(__as_ptr_smem(dstMem)), "l"(__as_ptr_gmem(srcMem)), "r"(copySize), "r"(__as_ptr_smem(smemBar))
        : "memory");
#endif
}

__device__ __forceinline__ void cp_async_bulk_s2g(void* dstMem, void const* srcMem, int copySize)
{
#if defined(__CUDACC__) || __CUDA_ARCH__ >= 900
    asm("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
        :
        : "l"(__as_ptr_gmem(dstMem)), "r"(__as_ptr_smem(srcMem)), "r"(copySize)
        : "memory");
#endif
}

__device__ __forceinline__ void cp_async_bulk_commit_group()
{
#if defined(__CUDACC__) || __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.commit_group;" : : :);
#endif
}

template <int N = 0>
__device__ __forceinline__ void cp_async_bulk_wait_group()
{
#if defined(__CUDACC__) || __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.wait_group %0;" : : "n"(N) : "memory");
#endif
}

template <int N = 0>
__device__ __forceinline__ void cp_async_bulk_wait_group_read()
{
#if defined(__CUDACC__) || __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.wait_group.read %0;" : : "n"(N) : "memory");
#endif
}

__host__ void MoeCommFieldInfo::fillFieldInfo(uint8_t* dataPtr, size_t elementSize, int vectorSize, int stride)
{
    TLLM_CHECK(elementSize == 1 || elementSize == 2 || elementSize == 4 || elementSize == 8 || elementSize == 16);

    dataPtrBase = dataPtr;

    uint64_t dataPtrU64 = reinterpret_cast<uint64_t>(dataPtr);

    while (elementSize < 16 && dataPtrU64 % (elementSize * 2) == 0 && vectorSize % 2 == 0 && stride % 2 == 0)
    {
        elementSize *= 2;
        vectorSize /= 2;
        stride /= 2;
    }

    if (elementSize == 16)
    {
        alignedUnitBit = 4;
    }
    else if (elementSize == 8)
    {
        alignedUnitBit = 3;
    }
    else if (elementSize == 4)
    {
        alignedUnitBit = 2;
    }
    else if (elementSize == 2)
    {
        alignedUnitBit = 1;
    }
    else
    {
        alignedUnitBit = 0;
    }

    alignedUnitCount = vectorSize;
    alignedUnitStride = stride;
}

void FusedMoeFieldInfo::fillFieldPlacementInfo(int topK, bool hasBasicFields)
{
    int basicFieldSize = 0;
    if (hasBasicFields)
    {
        basicFieldSize = topK * sizeof(int) + (expertScales != nullptr ? topK * sizeof(float) : 0);
        // align to 16 bytes
        basicFieldSize = (basicFieldSize + MoeCommFieldInfo::BYTES_PER_16B_BLOCK - 1)
            / MoeCommFieldInfo::BYTES_PER_16B_BLOCK * MoeCommFieldInfo::BYTES_PER_16B_BLOCK;
    }
    int offset = basicFieldSize;
    int unalignedFieldIndex = 0;
    for (int i = 0; i < fieldCount; i++)
    {
        fieldsInfo[i].packed16BOffset = offset / MoeCommFieldInfo::BYTES_PER_16B_BLOCK;
        offset += fieldsInfo[i].getFieldPackedSize();
        fieldsInfo[i].unalignedFieldIndex = unalignedFieldIndex;
        if (fieldsInfo[i].alignedUnitBit < 4)
        {
            unalignedFieldIndex++;
        }
    }
}

// #define DEBUG_PRINT

namespace fused_moe_impl
{

// returns copy size for txCount
__device__ __forceinline__ int startFieldG2S(MoeCommFieldInfo const& fieldInfo, int dataIndex,
    uint8_t* sharedMemoryBase, int warpId, int laneId, uint64_t* smemBar)
{
    // we can copy more data than needed, just align to 16 bytes.
    int alignedShmLoadOffset = fieldInfo.getUnpackedShmOffset();
    uint8_t* sharedMemoryLoadPtr = sharedMemoryBase + alignedShmLoadOffset;
    int copyByteCount = 0;
    uint8_t* loadPtr = fieldInfo.get16BAlignedLoadCopyRange(dataIndex, &copyByteCount);
    if (laneId == 0)
    {
#ifdef DEBUG_PRINT
        printf("warpId=%d, blockIdx.x=%d, in startFieldG2S alignedShmLoadOffset=%d, copyByteCount=%d\n", warpId,
            blockIdx.x, alignedShmLoadOffset, copyByteCount);
#endif
        cp_async_bulk_g2s(sharedMemoryLoadPtr, loadPtr, copyByteCount, smemBar);
    }
    return copyByteCount;
}

__device__ __forceinline__ void startFieldS2G(
    MoeCommFieldInfo const& fieldInfo, int dataIndex, uint8_t* sharedMemoryBase, int warpId, int laneId)
{
    int alignedShmStoreOffset = fieldInfo.getUnpackedShmOffset();
    uint8_t* sharedMemoryStorePtr = sharedMemoryBase + alignedShmStoreOffset;
    int copyByteCount = 0;
    int headTailShmIdx;
    int headTailGlobalIdx;
    uint8_t* storePtr
        = fieldInfo.get16BAlignedStoreCopyRange(dataIndex, &copyByteCount, laneId, &headTailShmIdx, &headTailGlobalIdx);
    if (copyByteCount > 0 && laneId == 0)
    {
#ifdef DEBUG_PRINT
        printf(
            "startFieldS2G, alignedShmStoreOffset=%d, blockIdx.x=%d, warpId=%d, laneId=%d, copy aligned "
            "copyByteCount=%d bytes.\n",
            alignedShmStoreOffset, blockIdx.x, warpId, laneId, copyByteCount);
#endif
        cp_async_bulk_s2g(storePtr, sharedMemoryStorePtr + MoeCommFieldInfo::BYTES_PER_16B_BLOCK, copyByteCount);
    }
#ifdef DEBUG_PRINT
    printf("startFieldS2G, blockIdx.x=%d, warpId=%d, laneId=%d, headTailShmIdx=%d to headTailGlobalIdx=%d\n",
        blockIdx.x, warpId, laneId, headTailShmIdx, headTailGlobalIdx);
#endif
    if (headTailGlobalIdx >= 0)
    {
#ifdef DEBUG_PRINT
        printf(
            "startFieldS2G, blockIdx.x=%d, warpId=%d, laneId=%d, copy headtail from headTailShmIdx=%d to "
            "headTailGlobalIdx=%d\n",
            blockIdx.x, warpId, laneId, headTailShmIdx, headTailGlobalIdx);
#endif
        // copy head and tail
        fieldInfo.getRawPtr(dataIndex, nullptr)[headTailGlobalIdx] = sharedMemoryStorePtr[headTailShmIdx];
    }
    __syncwarp();
}

// SRC_AFTER_DST is true, if src > dst, pack will use this,
// SRC_AFTER_DST is false, if src < dst, unpack will use this
template <typename T, bool SRC_AFTER_DST = true>
__device__ __forceinline__ void memmoveSharedMemory(uint8_t* dst, uint8_t const* src, int copySize, int laneId)
{
    int count = (copySize + sizeof(T) - 1) / sizeof(T);
    int warpLoopStart = SRC_AFTER_DST ? 0 : (count + WARP_SIZE - 1) / WARP_SIZE - 1;
    int warpLoopEnd = SRC_AFTER_DST ? (count + WARP_SIZE - 1) / WARP_SIZE : -1;
    int warpLoopUpdate = SRC_AFTER_DST ? 1 : -1;
    for (int i = warpLoopStart; i != warpLoopEnd; i += warpLoopUpdate)
    {
        int idx = laneId + i * WARP_SIZE;
        T data = T{};
        if (idx < count)
        {
            data = reinterpret_cast<T const*>(src)[idx];
        }
        __syncwarp();
        if (idx < count)
        {
            reinterpret_cast<T*>(dst)[idx] = data;
        }
        __syncwarp();
    }
}

template <bool IS_PACK = true>
__device__ __forceinline__ void memmoveFieldOnSharedMemory(
    MoeCommFieldInfo const& fieldInfo, int dataIndex, uint8_t* sharedMemoryBase, int laneId)
{
    int movOffset = fieldInfo.getMemmoveOffsets(dataIndex);
#ifdef DEBUG_PRINT
    if (laneId == 0)
    {
        printf("pack=%d, dataIndex=%d, movOffset=%d\n", IS_PACK ? 1 : 0, dataIndex, movOffset);
    }
#endif
    if (movOffset == 0)
    {
        // if movOffset is 0, src and dst are the same, don't need memmove.
        return;
    }
    int alignedBytes = 1 << fieldInfo.alignedUnitBit;
    int copySize = fieldInfo.alignedUnitCount * alignedBytes;
    uint8_t* sharedMemoryPacked = sharedMemoryBase + fieldInfo.getPackedShmOffset();
    uint8_t* sharedMemoryUnpacked = sharedMemoryPacked + movOffset;
    uint8_t* sharedMemoryDst = IS_PACK ? sharedMemoryPacked : sharedMemoryUnpacked;
    uint8_t* sharedMemorySrc = IS_PACK ? sharedMemoryUnpacked : sharedMemoryPacked;

    if (movOffset % 16 == 0)
    {
        memmoveSharedMemory<int4, IS_PACK>(sharedMemoryDst, sharedMemorySrc, copySize, laneId);
    }
    else if (movOffset % 8 == 0)
    {
        memmoveSharedMemory<int64_t, IS_PACK>(sharedMemoryDst, sharedMemorySrc, copySize, laneId);
    }
    else if (movOffset % 4 == 0)
    {
        memmoveSharedMemory<int, IS_PACK>(sharedMemoryDst, sharedMemorySrc, copySize, laneId);
    }
    else if (movOffset % 2 == 0)
    {
        memmoveSharedMemory<int16_t, IS_PACK>(sharedMemoryDst, sharedMemorySrc, copySize, laneId);
    }
    else
    {
        memmoveSharedMemory<int8_t, IS_PACK>(sharedMemoryDst, sharedMemorySrc, copySize, laneId);
    }
}

__device__ __forceinline__ void packAllFields(
    FusedMoeFieldInfo const& sendFieldInfo, int dataIndex, uint8_t* sharedMemoryBase, int laneId)
{
    for (int i = 0; i < sendFieldInfo.fieldCount; i++)
    {
        memmoveFieldOnSharedMemory<true>(sendFieldInfo.fieldsInfo[i], dataIndex, sharedMemoryBase, laneId);
    }
}

__device__ __forceinline__ void unpackAllFields(
    FusedMoeFieldInfo const& recvFieldInfo, int dataIndex, uint8_t* sharedMemoryBase, int laneId)
{
    for (int i = recvFieldInfo.fieldCount - 1; i >= 0; i--)
    {
        memmoveFieldOnSharedMemory<false>(recvFieldInfo.fieldsInfo[i], dataIndex, sharedMemoryBase, laneId);
    }
}

__device__ __forceinline__ void fixInvalidData(
    int* aligned128BytesShm, int countIn128Bytes, int fifoEntry128ByteIndexBase, int laneId)
{
    for (int idx = laneId; idx < countIn128Bytes; idx += WARP_SIZE)
    {
        int indexInFifoEntry = fifoEntry128ByteIndexBase + idx;
        int value0 = aligned128BytesShm[idx * MoeCommFieldInfo::INTS_PER_128B_BLOCK + indexInFifoEntry % WARP_SIZE];
        int value1
            = aligned128BytesShm[idx * MoeCommFieldInfo::INTS_PER_128B_BLOCK + (indexInFifoEntry + 1) % WARP_SIZE];
        if (value0 == FusedMoeCommunicator::INVALID_VALUE)
        {
            aligned128BytesShm[idx * MoeCommFieldInfo::INTS_PER_128B_BLOCK + indexInFifoEntry % WARP_SIZE]
                = FusedMoeCommunicator::FIXED_VALUE;
        }
        if (value1 == FusedMoeCommunicator::INVALID_VALUE)
        {
            aligned128BytesShm[idx * MoeCommFieldInfo::INTS_PER_128B_BLOCK + (indexInFifoEntry + 1) % WARP_SIZE]
                = FusedMoeCommunicator::FIXED_VALUE;
        }
    }
}

template <bool USE_FINISH = true>
__device__ __forceinline__ int dataReceivedInShm(
    int* aligned128BytesShm, int countIn128Bytes, int fifoEntry128ByteIndexBase, int laneId)
{
    // return value should be how many package already been received.
    // 0 means no data received, -1 means has received finish package(should be the very first 128 Byte).
    int totalValidCount = 0;
    for (int idxBase = 0; idxBase < countIn128Bytes; idxBase += WARP_SIZE)
    {
        int idx = idxBase + laneId;
        bool valid = false;
        bool finish = false;
        if (idx < countIn128Bytes)
        {
            int indexInFifoEntry = fifoEntry128ByteIndexBase + idx;
            int value0 = aligned128BytesShm[idx * MoeCommFieldInfo::INTS_PER_128B_BLOCK + indexInFifoEntry % WARP_SIZE];
            int value1
                = aligned128BytesShm[idx * MoeCommFieldInfo::INTS_PER_128B_BLOCK + (indexInFifoEntry + 1) % WARP_SIZE];
            valid = (value0 != FusedMoeCommunicator::INVALID_VALUE) && (value1 != FusedMoeCommunicator::INVALID_VALUE);
            if (USE_FINISH)
            {
                finish = (value0 != FusedMoeCommunicator::INVALID_VALUE)
                    && (value1 == FusedMoeCommunicator::INVALID_VALUE);
            }
        }
        __syncwarp();
        unsigned validMask = __ballot_sync(WARP_MASK, valid);
        // here we check valid in order, if previous valid is not true, we ignore the current valid.
        int validCount = (validMask == WARP_MASK) ? WARP_SIZE : (__ffs(~validMask) - 1);
        if (USE_FINISH)
        {
            unsigned finishedMask = __ballot_sync(WARP_MASK, finish);
            // finish should be the very first 128 Byte.
            if (finishedMask & 0x1)
            {
                return -1;
            }
        }
        totalValidCount += validCount;
        if (validCount != WARP_SIZE)
        {
            break;
        }
    }
    return totalValidCount;
}

__device__ __forceinline__ void g2sBasicFields(FusedMoeFieldInfo const& sendFieldInfo,
    MoeExpertParallelInfo const& expertParallelInfo, int dataIndex, uint8_t* sharedMemoryBase, int laneId)
{
    int topK = expertParallelInfo.topK;
    int* tokenSelectedSlotsPtr = sendFieldInfo.getTokenSelectedSlotsPtr(dataIndex, laneId, topK);
    float* scalePtr = sendFieldInfo.getScalePtr(dataIndex, laneId, topK);
    ldgsts<4>(reinterpret_cast<int*>(sharedMemoryBase) + laneId, tokenSelectedSlotsPtr, laneId < topK);
    ldgsts<4>(reinterpret_cast<int*>(sharedMemoryBase) + laneId + topK, reinterpret_cast<int*>(scalePtr),
        laneId < topK && sendFieldInfo.expertScales != nullptr);
}

// May commit 1 group for basic fields(tokenSelectedSlots and scales) if HAS_BASIC_FIELDS is true
// For other fields, use smemBar.
template <bool HAS_BASIC_FIELDS = true>
__device__ __forceinline__ uint32_t g2sAllFields(FusedMoeFieldInfo const& sendFieldInfo,
    MoeExpertParallelInfo const& expertParallelInfo, int dataIndex, uint8_t* sharedMemoryBase, int warpId, int laneId,
    uint64_t* smemBar)
{
    if (HAS_BASIC_FIELDS)
    {
#ifdef DEBUG_PRINT
        if (laneId == 0)
        {
            printf("warpId=%d, blockIdx.x=%d, in g2sAllFields before g2sBasicFields\n", warpId, blockIdx.x);
        }
#endif
        g2sBasicFields(sendFieldInfo, expertParallelInfo, dataIndex, sharedMemoryBase, laneId);
        cp_async_commit_group();
    }
    int asyncLoadSize = 0;
    for (int i = 0; i < sendFieldInfo.fieldCount; i++)
    {
#ifdef DEBUG_PRINT
        if (laneId == 0)
        {
            printf("warpId=%d, blockIdx.x=%d, in g2sAllFields before startFieldG2S for field %d, asyncLoadSize=%d\n",
                warpId, blockIdx.x, i, asyncLoadSize);
        }
#endif
        asyncLoadSize
            += startFieldG2S(sendFieldInfo.fieldsInfo[i], dataIndex, sharedMemoryBase, warpId, laneId, smemBar);
#ifdef DEBUG_PRINT
        if (laneId == 0)
        {
            printf("warpId=%d, blockIdx.x=%d, in g2sAllFields after startFieldG2S for field %d,  asyncLoadSize=%d\n",
                warpId, blockIdx.x, i, asyncLoadSize);
        }
#endif
    }
    return mbarrier_arrive_expect_tx(smemBar, laneId == 0 ? asyncLoadSize : 0);
}

__device__ __forceinline__ void s2gBasicFields(FusedMoeFieldInfo const& recvFieldInfo,
    MoeExpertParallelInfo const& expertParallelInfo, int dataIndex, uint8_t* sharedMemoryBase, int warpId, int laneId)
{
    int topK = expertParallelInfo.topK;
    int* tokenSelectedSlotsPtr = recvFieldInfo.getTokenSelectedSlotsPtr(dataIndex, laneId, topK);
    float* scalePtr = recvFieldInfo.getScalePtr(dataIndex, laneId, topK);
    if (laneId < topK)
    {
        int selectedSlot = reinterpret_cast<int*>(sharedMemoryBase)[laneId];
        *tokenSelectedSlotsPtr = selectedSlot;
        if (recvFieldInfo.expertScales != nullptr)
        {
            float scale = reinterpret_cast<float*>(sharedMemoryBase)[laneId + topK];
            *scalePtr = scale;
        }
    }
}

// Will commit 1 group, for all non-basic fields
template <bool HAS_BASIC_FIELDS = true>
__device__ __forceinline__ void s2gAllFields(FusedMoeFieldInfo const& recvFieldInfo,
    MoeExpertParallelInfo const& expertParallelInfo, int dataIndex, uint8_t* sharedMemoryBase, int warpId, int laneId)
{
    if (HAS_BASIC_FIELDS)
    {
        s2gBasicFields(recvFieldInfo, expertParallelInfo, dataIndex, sharedMemoryBase, warpId, laneId);
        __syncwarp();
    }
    for (int i = 0; i < recvFieldInfo.fieldCount; i++)
    {
        startFieldS2G(recvFieldInfo.fieldsInfo[i], dataIndex, sharedMemoryBase, warpId, laneId);
    }
    cp_async_bulk_commit_group();
}

__device__ __forceinline__ void initSmemBar(uint64_t* smemBar, int laneId)
{
    if (laneId == 0)
    {
        mbarrier_init(smemBar, WARP_SIZE);
    }
    __syncwarp();
}

__device__ __forceinline__ void smemBarWait(uint64_t* smemBar, uint32_t state)
{
    while (!mbarrier_try_wait_parity(smemBar, state))
    {
    }
}

} // namespace fused_moe_impl

namespace fused_moe_comm_tests
{

__global__ void g2sKernel(FusedMoeFieldInfo allFieldInfo, MoeExpertParallelInfo expertParallelInfo,
    MoeSingleCommMeta singleCommMeta, int tokenCount, int* shmDump, bool hasBasicFields)
{
    __shared__ uint64_t allWarpSmemBar[32];
    extern __shared__ int4 allWarpShm[];
    int laneId = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    int warpCount = blockDim.x / WARP_SIZE;
    int tokenIndex = warpId + blockIdx.x * warpCount;
    if (tokenIndex >= tokenCount)
    {
        return;
    }

#ifdef DEBUG_PRINT
    if (laneId == 0)
    {
        printf("warpId=%d, blockIdx.x=%d, topK=%d, starting g2sKernel for token=%d\n", warpId, blockIdx.x,
            expertParallelInfo.topK, tokenIndex);
    }
#endif

    tensorrt_llm::kernels::fused_moe_impl::initSmemBar(&allWarpSmemBar[warpId], laneId);

    uint8_t* sharedMemoryBase
        = reinterpret_cast<uint8_t*>(allWarpShm) + singleCommMeta.singleUnpackedAlignedSize * warpId;

    uint32_t waitToken;

    if (hasBasicFields)
    {
        waitToken = tensorrt_llm::kernels::fused_moe_impl::g2sAllFields<true>(
            allFieldInfo, expertParallelInfo, tokenIndex, sharedMemoryBase, warpId, laneId, &allWarpSmemBar[warpId]);
        cp_async_wait_group<0>();
        __syncwarp();
    }
    else
    {
        waitToken = tensorrt_llm::kernels::fused_moe_impl::g2sAllFields<false>(
            allFieldInfo, expertParallelInfo, tokenIndex, sharedMemoryBase, warpId, laneId, &allWarpSmemBar[warpId]);
    }

#ifdef DEBUG_PRINT
    if (laneId == 0)
    {
        printf("warpId=%d, blockIdx.x=%d, before smemBarWait token=%u\n", warpId, blockIdx.x, waitToken);
    }
#endif
    tensorrt_llm::kernels::fused_moe_impl::smemBarWait(&allWarpSmemBar[warpId], waitToken);
#ifdef DEBUG_PRINT
    if (laneId == 0)
    {
        printf("warpId=%d, blockIdx.x=%d, after smemBarWait token=%u\n", warpId, blockIdx.x, waitToken);
    }
#endif

    for (int offset = laneId; offset < singleCommMeta.singleUnpackedAlignedSize / sizeof(int); offset += WARP_SIZE)
    {
        shmDump[tokenIndex * singleCommMeta.singleUnpackedAlignedSize / sizeof(int) + offset]
            = reinterpret_cast<int*>(sharedMemoryBase)[offset];
    }
}

void launchSingleG2S(FusedMoeFieldInfo const& sendFieldInfo, MoeExpertParallelInfo const& expertParallelInfo,
    int tokenCount, int* shmDump, int warpsPerBlock, bool hasBasicFields, cudaStream_t stream)
{
    int warpShmSize = sendFieldInfo.computeSingleWarpShmSize(
        expertParallelInfo.topK, sendFieldInfo.expertScales != nullptr, hasBasicFields);
    dim3 blockDim(WARP_SIZE * warpsPerBlock, 1, 1);
    dim3 gridDim((tokenCount + warpsPerBlock - 1) / warpsPerBlock, 1, 1);
    MoeSingleCommMeta singleCommMeta;
    sendFieldInfo.fillMetaInfo(
        &singleCommMeta, expertParallelInfo.topK, sendFieldInfo.expertScales != nullptr, hasBasicFields);
    TLLM_CUDA_CHECK(
        cudaFuncSetAttribute(g2sKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, warpShmSize * warpsPerBlock));
    g2sKernel<<<gridDim, blockDim, warpShmSize * warpsPerBlock, stream>>>(
        sendFieldInfo, expertParallelInfo, singleCommMeta, tokenCount, shmDump, hasBasicFields);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

__global__ void s2gKernel(FusedMoeFieldInfo recvFieldInfo, MoeExpertParallelInfo expertParallelInfo,
    MoeSingleCommMeta singleCommMeta, int tokenCount, int* shmPreload, bool hasBasicFields)
{
    extern __shared__ int4 allWarpShm[];
    int laneId = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    int warpCount = blockDim.x / WARP_SIZE;
    int tokenIndex = warpId + blockIdx.x * warpCount;
    if (tokenIndex >= tokenCount)
    {
        return;
    }
    uint8_t* sharedMemoryBase
        = reinterpret_cast<uint8_t*>(allWarpShm) + singleCommMeta.singleUnpackedAlignedSize * warpId;

    for (int offset = laneId; offset < singleCommMeta.singleUnpackedAlignedSize / sizeof(int); offset += WARP_SIZE)
    {
        reinterpret_cast<int*>(sharedMemoryBase)[offset]
            = shmPreload[tokenIndex * singleCommMeta.singleUnpackedAlignedSize / sizeof(int) + offset];
    }
    __syncwarp();

    if (hasBasicFields)
    {
        tensorrt_llm::kernels::fused_moe_impl::s2gAllFields<true>(
            recvFieldInfo, expertParallelInfo, tokenIndex, sharedMemoryBase, warpId, laneId);
    }
    else
    {
        tensorrt_llm::kernels::fused_moe_impl::s2gAllFields<false>(
            recvFieldInfo, expertParallelInfo, tokenIndex, sharedMemoryBase, warpId, laneId);
    }

    cp_async_bulk_wait_group_read<0>();
    __syncwarp();
}

void launchSingleS2G(FusedMoeFieldInfo const& recvFieldInfo, MoeExpertParallelInfo const& expertParallelInfo,
    int tokenCount, int* shmPreload, int warpsPerBlock, bool hasBasicFields, cudaStream_t stream)
{
    int warpShmSize = recvFieldInfo.computeSingleWarpShmSize(
        expertParallelInfo.topK, recvFieldInfo.expertScales != nullptr, hasBasicFields);
    dim3 blockDim(WARP_SIZE * warpsPerBlock, 1, 1);
    dim3 gridDim((tokenCount + warpsPerBlock - 1) / warpsPerBlock, 1, 1);
    MoeSingleCommMeta singleCommMeta;
    recvFieldInfo.fillMetaInfo(
        &singleCommMeta, expertParallelInfo.topK, recvFieldInfo.expertScales != nullptr, hasBasicFields);
    TLLM_CUDA_CHECK(
        cudaFuncSetAttribute(s2gKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, warpShmSize * warpsPerBlock));
    s2gKernel<<<gridDim, blockDim, warpShmSize * warpsPerBlock, stream>>>(
        recvFieldInfo, expertParallelInfo, singleCommMeta, tokenCount, shmPreload, hasBasicFields);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

__global__ void loopbackKernel(FusedMoeFieldInfo sendFieldInfo, FusedMoeFieldInfo recvFieldInfo,
    MoeExpertParallelInfo expertParallelInfo, MoeSingleCommMeta sendCommMeta, MoeSingleCommMeta recvCommMeta,
    int* recvIndexMapping, int tokenCount, bool hasBasicFields)
{
    __shared__ uint64_t allWarpSmemBar[32];
    extern __shared__ int4 allWarpShm[];
    int laneId = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    int warpCount = blockDim.x / WARP_SIZE;
    int tokenIndex = warpId + blockIdx.x * warpCount;
    if (tokenIndex >= tokenCount)
    {
        return;
    }

    int recvTokenIndex = recvIndexMapping[tokenIndex];

#ifdef DEBUG_PRINT
    if (laneId == 0)
    {
        printf("warpId=%d, blockIdx.x=%d, topK=%d, starting loopbackKernel for token=%d\n", warpId, blockIdx.x,
            expertParallelInfo.topK, tokenIndex);
    }
#endif

    tensorrt_llm::kernels::fused_moe_impl::initSmemBar(&allWarpSmemBar[warpId], laneId);

    uint8_t* sharedMemoryBase
        = reinterpret_cast<uint8_t*>(allWarpShm) + sendCommMeta.singleUnpackedAlignedSize * warpId;

    uint32_t waitToken;

    if (hasBasicFields)
    {
        waitToken = tensorrt_llm::kernels::fused_moe_impl::g2sAllFields<true>(
            sendFieldInfo, expertParallelInfo, tokenIndex, sharedMemoryBase, warpId, laneId, &allWarpSmemBar[warpId]);
        cp_async_wait_group<0>();
        __syncwarp();
    }
    else
    {
        waitToken = tensorrt_llm::kernels::fused_moe_impl::g2sAllFields<false>(
            sendFieldInfo, expertParallelInfo, tokenIndex, sharedMemoryBase, warpId, laneId, &allWarpSmemBar[warpId]);
    }

    tensorrt_llm::kernels::fused_moe_impl::smemBarWait(&allWarpSmemBar[warpId], waitToken);

    tensorrt_llm::kernels::fused_moe_impl::packAllFields(sendFieldInfo, tokenIndex, sharedMemoryBase, laneId);

#ifdef DEBUG_PRINT
    for (int i = laneId; i < sendCommMeta.singlePackedAlignedSize / sizeof(int); i += WARP_SIZE)
    {
        printf("blockIdx.x=%d, warpId=%d, packed[%d] = %x\n", blockIdx.x, warpId, i,
            reinterpret_cast<int*>(&sharedMemoryBase[0])[i]);
    }
#endif

    tokenIndex = recvTokenIndex; // switch to recvTokenIndex;

    tensorrt_llm::kernels::fused_moe_impl::unpackAllFields(recvFieldInfo, tokenIndex, sharedMemoryBase, laneId);

#ifdef DEBUG_PRINT
    for (int i = laneId; i < recvCommMeta.singleUnpackedAlignedSize / sizeof(int); i += WARP_SIZE)
    {
        printf("blockIdx.x=%d, warpId=%d, unpacked[%d] = %x\n", blockIdx.x, warpId, i,
            reinterpret_cast<int*>(&sharedMemoryBase[0])[i]);
    }
#endif

    if (hasBasicFields)
    {
        tensorrt_llm::kernels::fused_moe_impl::s2gAllFields<true>(
            recvFieldInfo, expertParallelInfo, tokenIndex, sharedMemoryBase, warpId, laneId);
    }
    else
    {
        tensorrt_llm::kernels::fused_moe_impl::s2gAllFields<false>(
            recvFieldInfo, expertParallelInfo, tokenIndex, sharedMemoryBase, warpId, laneId);
    }

    cp_async_bulk_wait_group_read<0>();
    __syncwarp();
}

// G2S -> Pack -> Unpack -> S2G
void launchLoopback(FusedMoeFieldInfo const& sendFieldInfo, FusedMoeFieldInfo const& recvFieldInfo,
    MoeExpertParallelInfo const& expertParallelInfo, int* recvIndexMapping, int tokenCount, int warpsPerBlock,
    bool hasBasicFields, cudaStream_t stream)
{
    int warpSendShmSize = sendFieldInfo.computeSingleWarpShmSize(
        expertParallelInfo.topK, sendFieldInfo.expertScales != nullptr, hasBasicFields);
    int warpRecvShmSize = recvFieldInfo.computeSingleWarpShmSize(
        expertParallelInfo.topK, recvFieldInfo.expertScales != nullptr, hasBasicFields);
    int warpShmSize = warpSendShmSize;
    TLLM_CHECK_WITH_INFO(warpSendShmSize == warpRecvShmSize, "warpSendShmSize(%d) not same as warpRecvShmSize(%d)",
        warpSendShmSize, warpRecvShmSize);
    dim3 blockDim(WARP_SIZE * warpsPerBlock, 1, 1);
    dim3 gridDim((tokenCount + warpsPerBlock - 1) / warpsPerBlock, 1, 1);
    MoeSingleCommMeta sendCommMeta, recvCommMeta;
    sendFieldInfo.fillMetaInfo(
        &sendCommMeta, expertParallelInfo.topK, sendFieldInfo.expertScales != nullptr, hasBasicFields);
    recvFieldInfo.fillMetaInfo(
        &recvCommMeta, expertParallelInfo.topK, recvFieldInfo.expertScales != nullptr, hasBasicFields);
    TLLM_CUDA_CHECK(
        cudaFuncSetAttribute(loopbackKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, warpShmSize * warpsPerBlock));
    loopbackKernel<<<gridDim, blockDim, warpShmSize * warpsPerBlock, stream>>>(sendFieldInfo, recvFieldInfo,
        expertParallelInfo, sendCommMeta, recvCommMeta, recvIndexMapping, tokenCount, hasBasicFields);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace fused_moe_comm_tests

} // namespace kernels
} // namespace tensorrt_llm
