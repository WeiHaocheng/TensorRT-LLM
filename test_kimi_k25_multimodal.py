"""Quick test for Kimi K2.5 multimodal (image) inference."""
import os

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.inputs import default_multimodal_input_loader

MODEL_DIR = "/home/scratch.trt_llm_data_ci/llm-models/Kimi-K2.5-NVFP4/"
TEST_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "tests/integration/test_input_files")

test_cases = [
    {
        "prompt": "In as few words as possible, what city is this?",
        "media": os.path.join(TEST_IMAGES_DIR, "merlion.png"),
    },
    {
        "prompt": "Describe what you see in this image.",
        "media": os.path.join(TEST_IMAGES_DIR, "excel_table_test.jpg"),
    },
    {
        "prompt": "What sport is being played in this image?",
        "media":
        os.path.join(TEST_IMAGES_DIR,
                     "pexels-franco-monsalvo-252430633-32285228.jpg"),
    },
]


def main():
    from tensorrt_llm.llmapi import KvCacheConfig
    llm = LLM(
        model=MODEL_DIR,
        tensor_parallel_size=8,
        trust_remote_code=True,
        kv_cache_config=KvCacheConfig(enable_block_reuse=False),
    )
    sampling_params = SamplingParams(max_tokens=64)

    model_type = "kimi_k25"

    for i, tc in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test case {i}: {tc['prompt']}")
        print(f"Image: {tc['media']}")
        print(f"{'='*60}")

        inputs = default_multimodal_input_loader(
            tokenizer=llm.tokenizer,
            model_dir=str(llm._hf_model_dir),
            model_type=model_type,
            modality="image",
            prompts=[tc["prompt"]],
            media=[tc["media"]],
            image_data_format="pil",
            num_frames=8,
            device="cpu",
        )

        outputs = llm.generate(inputs, sampling_params)
        for out in outputs:
            generated = out.outputs[0].text
            print(f"Generated: {generated!r}")

    print("\n\nAll multimodal test cases completed successfully!")


if __name__ == "__main__":
    main()
