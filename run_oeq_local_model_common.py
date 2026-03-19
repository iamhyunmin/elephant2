import argparse
import os
import subprocess
import sys

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_dtype(dtype_name: str):
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_generation_components(model_name: str, dtype_name: str, device_map: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=resolve_dtype(dtype_name),
        device_map=device_map,
    )
    return tokenizer, model


def generate_one_response(tokenizer, model, prompt_text: str, max_new_tokens: int, temperature: float):
    model_inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {key: value.to(model.device) for key, value in model_inputs.items()}
    terminators = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is not None:
        terminators.append(eot_id)

    generate_kwargs = {
        **model_inputs,
        "max_new_tokens": max_new_tokens,
        "eos_token_id": terminators,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = 0.9

    with torch.no_grad():
        output = model.generate(**generate_kwargs)

    prompt_length = model_inputs["input_ids"].shape[-1]
    generated = output[0][prompt_length:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def ensure_parent_dir(path: str):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def load_or_initialize_output(input_df: pd.DataFrame, output_file: str, response_column: str):
    if os.path.exists(output_file):
        output_df = pd.read_csv(output_file)
        if response_column not in output_df.columns:
            output_df[response_column] = pd.NA
        return output_df

    output_df = input_df.copy()
    output_df[response_column] = pd.NA
    return output_df


def generate_responses(args):
    input_df = pd.read_csv(args.input_file)
    if args.input_column not in input_df.columns:
        raise ValueError(f"Input column '{args.input_column}' not found in {args.input_file}.")

    ensure_parent_dir(args.output_file)
    output_df = load_or_initialize_output(input_df, args.output_file, args.response_column)

    if len(output_df) != len(input_df):
        raise ValueError("Existing output file row count does not match the input dataset.")

    tokenizer, model = load_generation_components(
        args.model_name,
        args.dtype,
        args.device_map,
    )

    pending_indices = output_df.index[output_df[args.response_column].isna()].tolist()
    for idx in tqdm(pending_indices, desc=f"Generating {args.output_column_tag} responses"):
        prompt_text = str(output_df.at[idx, args.input_column])
        try:
            response_text = generate_one_response(
                tokenizer=tokenizer,
                model=model,
                prompt_text=prompt_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        except Exception as exc:
            response_text = f"[ERROR] {exc}"

        output_df.at[idx, args.response_column] = response_text

        if args.save_every and ((idx + 1) % args.save_every == 0):
            output_df.to_csv(args.output_file, index=False)

    output_df.to_csv(args.output_file, index=False)


def run_scorer(args):
    scorer_cmd = [
        sys.executable,
        "sycophancy_scorers.py",
        "--input_file",
        args.output_file,
        "--prompt_column",
        args.input_column,
        "--response_column",
        args.response_column,
        "--output_column_tag",
        args.output_column_tag,
        "--output_file",
        args.scored_output_file,
    ]

    if args.baseline is not None:
        scorer_cmd.extend(["--baseline", str(args.baseline)])

    if args.metrics:
        scorer_cmd.extend(args.metrics)

    subprocess.run(scorer_cmd, check=True)


def build_parser(description: str, default_model: str, default_output: str, default_response_column: str, default_tag: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model_name", default=default_model, help="HF model id or local model path.")
    parser.add_argument("--input_file", default="datasets/OEQ.csv", help="Path to the OEQ CSV.")
    parser.add_argument("--input_column", default="prompt", help="Column containing prompts.")
    parser.add_argument("--output_file", default=default_output, help="CSV path for generated responses.")
    parser.add_argument("--response_column", default=default_response_column, help="Column name to store model responses in.")
    parser.add_argument(
        "--scored_output_file",
        default=default_output.replace("_responses.csv", "_scored.csv"),
        help="CSV path for the scored outputs.",
    )
    parser.add_argument(
        "--output_column_tag",
        default=default_tag,
        help="Tag passed to sycophancy_scorers.py for output column names.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum tokens to generate per row.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model loading dtype. bfloat16 is a good default on H100.",
    )
    parser.add_argument("--device_map", default="auto", help="Transformers device_map to use when loading the model.")
    parser.add_argument("--save_every", type=int, default=25, help="Write intermediate generations every N completed rows.")
    parser.add_argument("--baseline", type=float, default=None, help="Optional baseline forwarded to sycophancy_scorers.py.")
    parser.add_argument(
        "--metrics",
        nargs="*",
        choices=["--validation", "--indirectness", "--framing"],
        default=["--validation", "--indirectness", "--framing"],
        help="Metric flags to pass to sycophancy_scorers.py.",
    )
    parser.add_argument("--skip_scoring", action="store_true", help="Only generate model responses and do not invoke sycophancy_scorers.py.")
    return parser


def run_pipeline(description: str, default_model: str, default_output: str, default_response_column: str, default_tag: str):
    parser = build_parser(description, default_model, default_output, default_response_column, default_tag)
    args = parser.parse_args()
    generate_responses(args)
    if not args.skip_scoring:
        run_scorer(args)
