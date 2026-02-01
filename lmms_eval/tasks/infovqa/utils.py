import json
import os
import re

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.api.metrics import levenshtein_distance


def infovqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def infovqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def infovqa_process_results(doc, results):
    pred = results[0].strip()
    
    # Robust cleanup
    pred = pred.replace("**", "").replace("*", "")
    match = re.search(r"(?:answer|option)(?: is| is:|:)?\s*(?:the)?\s*(.*)", pred, re.IGNORECASE)
    if match:
        pred = match.group(1)
    if pred.endswith("."):
        pred = pred[:-1]
    pred = pred.strip()
    
    # ANLS calculation
    answers = doc["answers"] if "answers" in doc else doc.get("answer", [])
    if isinstance(answers, str):
        answers = [answers]
        
    values = []
    for answer in answers:
        gt_answer = " ".join(answer.strip().lower().split())
        det_answer = " ".join(pred.strip().lower().split())

        dist = levenshtein_distance(gt_answer, det_answer)
        length = max(len(answer.upper()), len(pred.upper()))
        values.append(0.0 if length == 0 else float(dist) / float(length))

    anls_score = 1 - min(values) if values else 0.0
    if anls_score < 0.5:
        anls_score = 0.0
        
    return {"anls": anls_score}


def infovqa_test_process_results(doc, results):
    pred = results[0].strip()
    if pred.endswith("."):
        pred = pred[:-1]
    pred = pred.replace("*", "").strip()
    
    questionId = doc["questionId"]
    return {"submission": {"questionId": int(questionId), "answer": pred}}


def infovqa_test_aggregate_results(results, args):
    # save results as json
    file = generate_submission_file("infovqa_test_for_submission.json", args)
    with open(file, "w") as f:
        json.dump(results, f)
    eval_logger.info(f"Results saved to {file}")
