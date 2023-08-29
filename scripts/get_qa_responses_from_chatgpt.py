#!/usr/bin/env python3
"""Given a data file with questions and retrieval results to use, run longchat to get responses.

Currently, this script only supports `longchat-13b-16k`.

The retrieval results are used in the exact order that they're given.
"""
import argparse
import dataclasses
import json
import logging
import math
import os
import pathlib
import random
import sys
from copy import deepcopy

import openai
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
from xopen import xopen

from src.lost_in_the_middle.prompting import Document, get_closedbook_qa_prompt, get_qa_prompt

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger(__name__)
random.seed(0)

N_INPUTS = 100


def main(
    input_path,
    model_name,
    temperature,
    top_p,
    batch_size,
    closedbook,
    prompt_mention_random_ordering,
    use_random_ordering,
    query_aware_contextualization,
    num_gpus,
    max_memory_per_gpu,
    longchat_flash_attn,
    longchat_ratio,
    max_new_tokens,
    output_path,
):
    # Create directory for output path if it doesn't exist.
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    examples = []
    prompts = []
    all_model_documents = []
    did_format_warn = False

    # Fetch all of the prompts
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            # Get the prediction for the input example
            question = input_example["question"]
            if closedbook:
                documents = []
            else:
                documents = []
                for ctx in deepcopy(input_example["ctxs"]):
                    documents.append(Document.from_dict(ctx))
                if not documents:
                    raise ValueError(f"Did not find any documents for example: {input_example}")

            if use_random_ordering:
                # Randomly order only the distractors (isgold is False), keeping isgold documents
                # at their existing index.
                (original_gold_index,) = [idx for idx, doc in enumerate(documents) if doc.isgold is True]
                original_gold_document = documents[original_gold_index]
                distractors = [doc for doc in documents if doc.isgold is False]
                random.shuffle(distractors)
                distractors.insert(original_gold_index, original_gold_document)
                documents = distractors

            if closedbook:
                prompt = get_closedbook_qa_prompt(question)
            else:
                prompt = get_qa_prompt(
                    question,
                    documents,
                    mention_random_ordering=prompt_mention_random_ordering,
                    query_aware_contextualization=query_aware_contextualization,
                )

            prompts.append(prompt)
            examples.append(deepcopy(input_example))
            all_model_documents.append(documents)

    # Get responses for all of the prompts
    responses = []
    for inputs in tqdm(prompts[:N_INPUTS]):
        output = get_openai_chat_completion(
            model=model_name,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **inputs,
        )

        responses.append(output)

    with xopen(output_path, "w") as f:
        for example, model_documents, prompt, response in zip(examples, all_model_documents, prompts, responses):
            output_example = deepcopy(example)
            # Add some extra metadata to the output example
            output_example["model_prompt"] = prompt
            output_example["model_documents"] = [dataclasses.asdict(document) for document in model_documents]
            output_example["model_answer"] = response
            output_example["model"] = model_name
            output_example["model_temperature"] = temperature
            output_example["model_top_p"] = top_p
            output_example["model_prompt_mention_random_ordering"] = prompt_mention_random_ordering
            output_example["model_use_random_ordering"] = use_random_ordering
            f.write(json.dumps(output_example) + "\n")


def get_openai_chat_completion(model, temperature, top_p, max_tokens, system_message, user_message):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    except Exception as e:
        print(f"Error: {e}\nReturning empty string.")
        return ""

    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        help="Path to data with questions and documents to use.",
        default="qa_data/20_total_documents/nq-open-20_total_documents_gold_at_19.jsonl.gz",
    )
    parser.add_argument("--model", help="Model to use in generating responses", default="gpt-3.5-turbo-0613")
    parser.add_argument("--temperature", help="Temperature to use in generation", type=float, default=0.0)
    parser.add_argument("--top-p", help="Top-p to use in generation", type=float, default=1.0)
    parser.add_argument("--batch-size", help="Batch size use in generation", type=int, default=8)
    parser.add_argument(
        "--output-path",
        help="Path to write output file of generated responses",
        default="qa_predictions/nq-open-20_total_documents_gold_at_19-gpt-3.5-turbo-0613-predictions.jsonl.gz",
    )
    parser.add_argument("--num-gpus", help="Number of GPUs to use", type=int)
    parser.add_argument(
        "--closedbook", action="store_true", help="Run the model in closed-book mode (i.e., don't use documents)."
    )
    parser.add_argument(
        "--prompt-mention-random-ordering",
        action="store_true",
        help="Mention that search results are ordered randomly in the prompt",
    )
    parser.add_argument(
        "--use-random-ordering",
        action="store_true",
        help="Randomize the ordering of the distractors, rather than sorting by relevance.",
    )
    parser.add_argument(
        "--query-aware-contextualization",
        action="store_true",
        help="Place the question both before and after the documents.",
    )
    parser.add_argument(
        "--longchat-flash-attn",
        action="store_true",
        help="Only apply to longchat models. Whether to enable flash attention to save memory, but slower.",
    )
    parser.add_argument(
        "--longchat-ratio",
        type=int,
        default=8,
        help="Only apply to longchat models. Use ratio=8 for 16K context length model. Only ratio=8 is supported now.",
    )
    parser.add_argument(
        "--max-memory-per-gpu",
        help="Maximum memory to use per GPU (in GiB) for multi-device parallelism, e.g., 80",
        type=int,
    )
    parser.add_argument(
        "--max-new-tokens",
        help="Maximum number of new tokens to generate",
        type=int,
        default=100,
    )
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    main(
        args.input_path,
        args.model,
        args.temperature,
        args.top_p,
        args.batch_size,
        args.closedbook,
        args.prompt_mention_random_ordering,
        args.use_random_ordering,
        args.query_aware_contextualization,
        args.num_gpus,
        args.max_memory_per_gpu,
        args.longchat_flash_attn,
        args.longchat_ratio,
        args.max_new_tokens,
        args.output_path,
    )
    logger.info("finished running %s", sys.argv[0])
