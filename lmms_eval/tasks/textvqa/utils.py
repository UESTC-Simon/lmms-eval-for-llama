import datetime
import json
import re
import statistics
from typing import List

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


# ----------------------------
# Normalization helpers
# ----------------------------

# 前后常见标点/引号（包含中英文引号与撇号变体）
_LEAD_PUNCT = re.compile(r"""^[\s\(\[\{'"“”‘’]+""")
_TAIL_PUNCT = re.compile(r"""[\s\.\,\;\:\!\?\)\]\}'"“”‘’]+$""")

# 把各种 unicode 引号/撇号统一为 '
_QUOTE_MAP = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "‛": "'",
        "＇": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "″": '"',
    }
)

# 常见“答案前缀”
_ANSWER_PREFIX = re.compile(
    r"""(?is)^\s*(?:final\s+answer|answer|option)\s*(?:is|is:|:)?\s*(?:the\s+)?(.*)$"""
)

def _strip_edge_punct(s: str) -> str:
    s = s.strip()
    s = _LEAD_PUNCT.sub("", s)
    s = _TAIL_PUNCT.sub("", s)
    return s.strip()

def _fix_apostrophe_spacing(s: str) -> str:
    # 处理 "fry 's" -> "fry's" 以及 "don ' t" -> "don't"
    s = re.sub(r"\s*'\s*", "'", s)
    return s

def _cleanup_raw_answer(raw: str) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()

    # 去掉 markdown 强调符
    s = s.replace("**", "").replace("*", "")

    # 统一引号/撇号
    s = s.translate(_QUOTE_MAP)

    # 去掉 Answer: / The answer is 之类前缀（只取后半段）
    m = _ANSWER_PREFIX.match(s)
    if m:
        s = m.group(1).strip()

    # 多行只取第一条非空行（避免模型输出解释）
    if "\n" in s:
        for line in s.splitlines():
            line = line.strip()
            if line:
                s = line
                break

    # 去掉前后标点、合并空格、修复撇号空格
    s = _strip_edge_punct(s)
    s = re.sub(r"\s+", " ", s)
    s = _fix_apostrophe_spacing(s)
    s = s.strip()

    return s


def _canonical(processor: EvalAIAnswerProcessor, raw: str) -> str:
    """
    先做一层更鲁棒的清洗，再交给 EvalAIAnswerProcessor（其内部会做 lower、去冠词等 VQA 规范化）
    最后再补一层撇号空格修复/空格折叠，避免 processor 输出里出现 "fry 's" 这种情况。
    """
    s = _cleanup_raw_answer(raw)
    s = processor(s)
    s = s.translate(_QUOTE_MAP)
    s = _fix_apostrophe_spacing(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ----------------------------
# Task hooks
# ----------------------------

def textvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def textvqa_process_results(doc, result):
    """
    保持 TextVQA/VQA 共识打分：
      对每个 GT_i，统计其他 9 个 GT 中与 pred 相同的个数 m，
      acc_i = min(1, m/3)，最后取 mean(acc_i)
    只增强归一化鲁棒性，避免标点/引号/撇号空格导致的“假不匹配”。
    """
    processor = EvalAIAnswerProcessor()
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."

    pred = _canonical(processor, result[0])

    answers = doc.get("answers", None)
    if not answers:
        accuracy = 0.0
    else:
        # 不原地改写 doc["answers"]
        gt_list: List[str] = [_canonical(processor, a) for a in answers if a is not None]

        gtAcc = []
        for i in range(len(gt_list)):
            other = [gt_list[j] for j in range(len(gt_list)) if j != i]
            matches = sum(1 for a in other if a == pred)
            acc = min(1.0, float(matches) / 3.0)
            gtAcc.append(acc)

        accuracy = float(statistics.mean(gtAcc)) if gtAcc else 0.0

    return {
        "exact_match": accuracy,
        "submission": {
            "question_id": doc["question_id"],
            "answer": pred,
        },
    }


def textvqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = ""
    post_prompt = ""
    ocr_ref = ""

    if lmms_eval_specific_kwargs:
        if "pre_prompt" in lmms_eval_specific_kwargs:
            pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
        if "post_prompt" in lmms_eval_specific_kwargs:
            post_prompt = lmms_eval_specific_kwargs["post_prompt"]
        if "ocr" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["ocr"]:
            if "ocr_tokens" in doc and isinstance(doc["ocr_tokens"], list):
                ocr_ref = f"\nReference OCR token: {', '.join(doc['ocr_tokens'])}"

    # 不使用 capitalize()，避免改变原始问题（有些数据集大小写/标点也可能影响提示）
    return f"{pre_prompt}{doc.get('question','')}{ocr_ref}{post_prompt}"


def textvqa_aggregate_submissions(results, args):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path = generate_submission_file(f"textvqa_submission_{now_date_time}.json", args)
    with open(path, "w") as f:
        json.dump(results, f)
    eval_logger.info(f"Submission file saved to {path}")
