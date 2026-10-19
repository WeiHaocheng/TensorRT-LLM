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

#pragma once
#include "tensorrt_llm/common/config.h"
#include <limits>
#include <type_traits>

#include "cutlass/epilogue/thread/activation.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::cutlass_kernels
{
// ============================== Activation Adaptors =================================

template <template <class> class ActFn>
struct IdentityAdaptor
{
    constexpr static bool IS_GLU = false;
    float alpha = 1.0f;
    float beta = 0.0f;
    float limit = std::numeric_limits<float>::infinity();

    template <class T>
    __device__ T operator()(T const& x) const
    {
        ActFn<T> fn{};
        return fn(x);
    }
};

template <template <class> class ActFn>
struct GLUAdaptor
{
    constexpr static bool IS_GLU = true;
    float alpha = 1.0f;
    float beta = 0.0f;
    float limit = std::numeric_limits<float>::infinity();

    template <class T>
    __device__ T operator()(T const& gate, T const& linear) const
    {
        ActFn<T> fn{};
        return fn(gate) * linear;
    }
};

struct SwigluBiasAdaptor
{
    constexpr static bool IS_GLU = true;
    float alpha = 1.0f;
    float beta = 0.0f;
    float limit = std::numeric_limits<float>::infinity();
    bool clampAfterSilu = false;

    template <class T>
    __device__ T operator()(T const& gate, T const& linear) const
    {
        cutlass::epilogue::thread::Sigmoid<T> fn{};
        T linear_clamped = cutlass::maximum<T>{}(cutlass::minimum<T>{}(linear, limit), -limit);
        if (clampAfterSilu)
        {
            T gate_activated = gate * fn(gate * alpha);
            return cutlass::minimum<T>{}(gate_activated, limit) * (linear_clamped + beta);
        }
        T gate_clamped = cutlass::minimum<T>{}(gate, limit);
        return gate_clamped * fn(gate_clamped * alpha) * (linear_clamped + beta);
    }
};

struct SiTuAdaptor
{
    constexpr static bool IS_GLU = true;
    float alpha = 1.0f;
    float beta = 1.0f;
    float limit = std::numeric_limits<float>::infinity();

    template <class T>
    __device__ T operator()(T const& gate, T const& linear) const
    {
        cutlass::epilogue::thread::Sigmoid<T> sigmoid{};
        cutlass::epilogue::thread::Tanh<T> tanhFn{};
        return tanhFn(gate * (1.0f / alpha)) * alpha * sigmoid(gate) * tanhFn(linear * (1.0f / beta)) * beta;
    }
};

// Relu2 under a tanh soft cap: relu(limit * tanh(x / limit))^2. Non-gated. Selected by ActivationType::Relu2 with a
// per-expert ActivationParams::swiglu_limit (the cap); plain Relu2 keeps IdentityAdaptor<Relu2>. Computed in fp32,
// x / limit as x times fp32(1 / limit), tanh by cutlass::fast_tanh (tanh.approx.f32, relative error below 2^-10.9,
// an eighth of a bf16 ulp; the accurate tanhf would make the activation pass compute-bound), and rounded once where
// the caller stores the result.
struct SoftcapRelu2Adaptor
{
    constexpr static bool IS_GLU = false;
    float alpha = 1.0f;
    float beta = 0.0f;
    float limit = std::numeric_limits<float>::infinity();

    __device__ static float apply(float x, float cap, float inv_cap)
    {
        float const capped = cap * cutlass::fast_tanh(x * inv_cap);
        float const positive = capped > 0.0f ? capped : 0.0f;
        return positive * positive;
    }

    template <class T>
    __device__ T operator()(T const& x) const
    {
        float const inv_limit = 1.0f / limit;
        if constexpr (std::is_arithmetic_v<T>)
        {
            return static_cast<T>(apply(static_cast<float>(x), limit, inv_limit));
        }
        else
        {
            T out;
#pragma unroll
            for (int i = 0; i < static_cast<int>(T::kElements); ++i)
            {
                out[i] = static_cast<typename T::Element>(apply(static_cast<float>(x[i]), limit, inv_limit));
            }
            return out;
        }
    }
};

} // namespace kernels::cutlass_kernels

TRTLLM_NAMESPACE_END
