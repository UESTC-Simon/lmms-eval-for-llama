import re

from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.filters.transformation import MapFilter


def ai2d_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question, choices = doc["question"], doc["options"]
    len_choices = len(choices)
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if lmms_eval_specific_kwargs["prompt_format"] == "mcq":
        options = [chr(ord("A") + i) for i in range(len_choices)]
        choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
        return f"{pre_prompt}{question}\n{choices_str}{post_prompt}"
    elif lmms_eval_specific_kwargs["prompt_format"] == "qa":
        options = "\n".join(choices)
        return f"{pre_prompt}{question}{options}{post_prompt}"
    elif lmms_eval_specific_kwargs["prompt_format"] == "mcq_xcomposer":
        options = [chr(ord("A") + i) for i in range(len_choices)]
        choices_str = " ".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
        return f"{pre_prompt}{question}\nContext: N/A\n{choices_str}{post_prompt}"
    else:
        raise ValueError(f"Unknown prompt format: {lmms_eval_specific_kwargs['prompt_format']}")


def ai2d_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def ai2d_doc_to_target(doc, model_specific_target_kwargs):
    if model_specific_target_kwargs == "mcq":
        len_choices = len(doc["options"])
        options = [chr(ord("A") + i) for i in range(len_choices)]
        return options[int(doc["answer"])]
    elif model_specific_target_kwargs == "qa":
        return doc["options"][int(doc["answer"])]


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        filtered_resps = []

        for r, doc in zip(resps, docs):
            filtered = []
            for resp in r:
                resp = resp.strip()
                
                # Pattern 1: Starts with "X." or "X " or just "X"
                # Matches: "A", "A.", "A. The answer...", "A is the answer"
                match_start = re.match(r"^([A-D])(?:\.|,|\s|$|\))", resp, re.IGNORECASE)
                
                # Pattern 2: "The answer is X" or "The correct answer is X" or "Answer: X"
                # relaxed to match "answer is" followed optionally by "region" "option" etc, but let's keep it simple first
                match_phrase = re.search(r"(?:answer|option)(?: is| is:|:)?\s*(?:option)?\s*([A-D])\b", resp, re.IGNORECASE)
                
                # Pattern 3: Ends with "X." or "X"
                # Useful for "The correct option is D."
                match_end = re.search(r"\b([A-D])\.$", resp, re.IGNORECASE)

                if match_start:
                    filtered.append(match_start.group(1).upper())
                elif match_phrase:
                    filtered.append(match_phrase.group(1).upper())
                elif match_end:
                    filtered.append(match_end.group(1).upper())
                else: 
                     # Last resort: if the response is very short and contains a letter
                     clean_resp = resp.strip(" .()")
                     if len(clean_resp) == 1 and clean_resp.upper() in "ABCD":
                         filtered.append(clean_resp.upper())
                     else:
                        filtered.append(resp)

            filtered_resps.append(filtered[0])

        return filtered_resps
