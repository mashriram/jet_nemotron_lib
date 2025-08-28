# examples/quickstart.py

# This script requires the TensorRT-LLM library to be installed.
# See: https://nvidia.github.io/TensorRT-LLM/

# Note: This is a conceptual workflow. The model must first be created,
# fine-tuned, and saved using the 'run_finetuning.py' script.

try:
    from tensorrt_llm import LLM, SamplingParams
except ImportError:
    print(
        "TensorRT-LLM is not installed. This is required for production-level performance."
    )
    LLM = None
    SamplingParams = None


def run_production_inference(model_path: str, prompts: list[str]):
    """
    Deploys the fine-tuned hybrid model with TensorRT-LLM for high-performance inference.
    """
    if LLM is None:
        print("Cannot run inference without TensorRT-LLM.")
        return

    print("Loading optimized model with TensorRT-LLM...")
    # TensorRT-LLM's PyTorch backend can directly load and optimize a PyTorch model.
    llm_engine = LLM(model=model_path, tensor_parallel_size=1)

    sampling_params = (
        SamplingParams(max_tokens=256, temperature=0.7, top_p=0.9)
        if SamplingParams
        else None
    )

    print("Running inference...")
    outputs = llm_engine.generate(prompts, sampling_params)

    for output in outputs:
        print("-" * 50)
        print(f"Prompt: {output.prompt}")
        print(f"Generated Text: {output.outputs[0].text}")
        print("-" * 50)


if __name__ == "__main__":
    # This path must point to a model that has already been fine-tuned.
    fine_tuned_model_dir = "./jet_nemotron_finetuned"

    sample_prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a Python function to compute the Fibonacci sequence.",
    ]

    run_production_inference(fine_tuned_model_dir, sample_prompts)
