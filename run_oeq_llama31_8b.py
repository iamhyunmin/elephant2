from run_oeq_local_model_common import run_pipeline


if __name__ == "__main__":
    run_pipeline(
        description="Run OEQ generation with Llama 3.1 8B Instruct, then score outputs with ELEPHANT.",
        default_model="meta-llama/Llama-3.1-8B-Instruct",
        default_output="outputs/OEQ_llama31_8b_responses.csv",
        default_response_column="llama31_8b_response",
        default_tag="llama31_8b",
    )
