import dataclasses
import json
import logging
import os
import pathlib
import random
import statistics
from copy import deepcopy
import time

import matplotlib.pyplot as plt
from tqdm import tqdm
from xopen import xopen

from src.lost_in_the_middle.metrics import best_subspan_em
from src.lost_in_the_middle.prompting import Document, get_closedbook_qa_prompt, get_qa_prompt

from src.lost_in_the_middle.gpt import get_openai_chat_completion
from src.lost_in_the_middle.prompts.prompt_functions import interleaved_prompt, summarize_first

from src.lost_in_the_middle.metrics import best_subspan_em


plt.style.use("ggplot")

logger = logging.getLogger(__name__)
random.seed(0)

N_INPUTS = 100


METRICS = [
    (best_subspan_em, "best_subspan_em"),
]


def get_qa_responses(
    input_path="qa_data/20_total_documents/nq-open-20_total_documents_gold_at_19.jsonl.gz",
    model_name="gpt-3.5-turbo-0613",
    temperature=0.0,
    top_p=1.0,
    closedbook=False,
    use_random_ordering=False,
    max_new_tokens=100,
    output_path="qa_predictions/nq-open-20_total_documents_gold_at_19-gpt-3.5-turbo-0613-predictions.jsonl.gz",
    prompt_function=get_qa_prompt,
):
    """
    prompt_function: Callable(question, documents) -> {"system_Message": system_message, "user_message": user_message}
    """
    # Create directory for output path if it doesn't exist.
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    examples = []
    prompts = []
    all_model_documents = []

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
                prompt = prompt_function(
                    question,
                    documents,
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
        if prompt_function == summarize_first:
            new_system_message = "Write a high-quality answer for the given question using only the provided text"
            new_user_message = f"{output}\n\nQuestion: {question}\nAnswer:"
            output = get_openai_chat_completion(
                model=model_name,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                system_message=new_system_message,
                user_message=new_user_message,
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
            output_example["model_use_random_ordering"] = use_random_ordering
            f.write(json.dumps(output_example) + "\n")


def evaluate_qa_responses(
    input_path,
    output_path,
):
    def get_metrics_for_example(example):
        gold_answers = example["answers"]
        model_answer = example["model_answer"]

        # NOTE: we take everything up to the first newline, since otherwise models could hack
        # the metric by simply copying te input context (as the gold answer is guaranteed
        # to occur in the input context).
        model_answer = model_answer.split("\n")[0].strip()

        example_metrics = {}
        for metric, metric_name in METRICS:
            example_metrics[metric_name] = metric(prediction=model_answer, ground_truths=gold_answers)
        return (example_metrics, example)

    all_examples = []
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            all_examples.append(input_example)

    # Compute normal metrics in parallel, if applicable
    logger.info("Computing metrics")
    all_example_metrics = []
    for example in tqdm(all_examples):
        all_example_metrics.append(get_metrics_for_example(example))

    # Average metrics across examples

    for _, metric_name in METRICS:
        average_metric_value = statistics.mean(
            example_metrics[metric_name] for (example_metrics, _) in all_example_metrics
        )
        logger.info(f"{metric_name}: {average_metric_value}")

    if output_path:
        with xopen(output_path, "w") as f:
            for example_metrics, example in all_example_metrics:
                example_with_metrics = deepcopy(example)
                for metric_name, metric_value in example_metrics.items():
                    example_with_metrics[f"metric_{metric_name}"] = metric_value
                f.write(json.dumps(example_with_metrics) + "\n")

    return average_metric_value


def run_qa_experiment(prompt_function):
    results = {}
    function_name = prompt_function.__name__
    for gold_index in [0, 4, 9, 14, 19]:
        predictions_path = f"qa_predictions/{function_name}/nq-open-20_total_documents_gold_at_{gold_index}-gpt-3.5-turbo-0613-predictions.jsonl.gz"
        get_qa_responses(
            input_path=f"qa_data/20_total_documents/nq-open-20_total_documents_gold_at_{gold_index}.jsonl.gz",
            output_path=predictions_path,
            prompt_function=prompt_function,
        )
        metric_value = evaluate_qa_responses(
            input_path=predictions_path,
            output_path=f"qa_predictions/{function_name}/q-open-20_total_documents_gold_at_{gold_index}-gpt-3.5-turbo-0613-predictions-scored.jsonl.gz",
        )
        results[gold_index] = metric_value
    with open(f"qa_predictions/{function_name}/results.json", "w") as f:
        json.dump(results, f)
    return results


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)

    # results = run_qa_experiment(get_qa_prompt)
    # results = run_qa_experiment(interleaved_prompt)
    # results = run_qa_experiment(know_your_weakness)
    results = run_qa_experiment(summarize_first)
    
    fig, ax = plt.subplots()
    ax.plot(results.keys(), results.values(), marker="o")
    ax.set_xticks(list(results))
    ax.set_xticklabels([k + 1 for k in results.keys()])
    ax.set_xlabel("Position of Document with the Answer")
    ax.set_ylabel("Accuracy")
    ax.set_title("20 Total Retrieved Documents")
    fig.savefig("results.png")
    plt.close()

    previous_results = pd.DataFrame(
        [
            {0: 0.67, 4: 0.48, 9: 0.53, 14: 0.51, 19: 0.61},
            {0: 0.81, 4: 0.55, 9: 0.48, 14: 0.51, 19: 0.56},
            {0: 0.7, 4: 0.53, 9: 0.5, 14: 0.53, 19: 0.58},
        ]
    ).transpose()
    previous_results.columns = ["Original", "Interleaved", "Know your weakness"]
    
    fig, ax = plt.subplots()
    previous_results.plot(ax=ax, marker="o")
    ax.set_xticks(list(results))
    ax.set_xticklabels([k + 1 for k in results.keys()])
    ax.set_xlabel("Position of Document with the Answer")
    ax.set_ylabel("Accuracy")
    ax.set_title("20 Total Retrieved Documents")
    fig.savefig("previous_results.png")
    plt.close()