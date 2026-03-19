from run_oeq_local_model_common import run_pipeline


if __name__ == "__main__":
    run_pipeline(
        description="Run OEQ generation with Tulu 3.1 SFT, then score outputs with ELEPHANT.",
        default_model="allenai/Llama-3.1-Tulu-3-8B-SFT",
        default_output="outputs/OEQ_tulu31_sft_responses.csv",
        default_response_column="tulu31_sft_response",
        default_tag="tulu31_sft",
    )
