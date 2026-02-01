import re

def chartqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def chartqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def chartqa_process_results(doc, results):
    pred = results[0].strip()
    
    # Handle "Answer:" format (case-insensitive, handles **Answer:** etc)
    pred_lower = pred.lower()
    if "answer:" in pred_lower:
        pred = pred[pred_lower.rfind("answer:") + 7:]
    elif "answer is" in pred_lower:
        pred = pred[pred_lower.rfind("answer is") + 9:]
    
    # Only take the first line of the extracted part to avoid following explanations
    pred = pred.split("\n")[0]

    # Clean up the prediction
    pred = pred.strip()
    # Remove trailing period
    if pred.endswith("."):
        pred = pred[:-1]
    # Remove markdown bold/italic markers
    pred = pred.replace("*", "").strip()
    
    # Try to extract the number if "Number Unit" pattern prevents float conversion
    # Only if prediction starts with a number (ignoring currency/signs)
    # Check for currency prefix
    match_currency = re.match(r"^[\$€£¥]([\d\.,]+.*)$", pred)
    if match_currency:
        pred = match_currency.group(1)
        
    # Check for "Number Unit" pattern e.g. "42.5 million", "100 kg", "5 billion euros"
    # Matches data followed by text (must start with letter to avoid matching .3 in 1.2.3 as text)
    match_unit = re.match(r"^(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:[a-zA-Z]+.*)$", pred)
    if match_unit:
        pred = match_unit.group(1)
    
    # Special handling for percentage: ChartQA targets are often raw numbers (e.g., 75 for 75%).
    # relaxed_correctness divides by 100 if it sees %, which might fail here.
    # So we strip % aggressively to match the target's scale.
    if pred.endswith("%"):
        pred = pred[:-1].strip()
    
    # If the prediction is still a full sentence (no Answer: tag found or processed poorly),
    # and we expect a short answer, this might fail unless we do more aggressive extraction.
    # But relaxed_correctness handles some of it.

    type = doc["type"]
    score = relaxed_correctness(pred, doc["answer"])
    score = 1.0 if score else 0.0
    return_dict = {"relaxed_overall": score}
    if type == "human_test":
        return_dict["relaxed_human_split"] = score
    else:
        return_dict["relaxed_augmented_split"] = score
    return return_dict


def relaxed_correctness(prediction, target, max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    This funcion is taken from https://github.com/QwenLM/Qwen-VL/blob/34b4c0ee7b07726371b960911f249fe61b362ca3/eval_mm/evaluate_vqa.py#L113
    Args:
      target: List of target string.
      prediction: List of predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str):
        try:
            text = text.replace(",", "")
            if text.endswith("%"):
                # Convert percentages to floats.
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction = prediction.strip().rstrip(".")
    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()
