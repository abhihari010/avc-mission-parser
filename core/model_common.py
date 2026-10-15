"""
Keeps the three models on the same comparison rules. It prepares their differently
formatted prompts, reads their answers, verifies the frozen datasets, and calculates the
scores. The public workflow uses these helpers automatically.
"""
# Previously: comparison_common.py
import ast
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

from schema import FUNCTION_SCHEMA, FIELDS, build_messages, parse_function_call

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "functiongemma": "google/functiongemma-270m-it",
    "granite350": "ibm-granite/granite-4.0-350m",
    "lfm350": "LiquidAI/LFM2.5-350M",
}
LFM_MODELS = {"lfm350"}
JSON_TOOL_MODELS = {"granite350"}
MAX_NEW_TOKENS = 256
ENUMS = {f: FUNCTION_SCHEMA["function"]["parameters"]["properties"][f]["enum"] for f in FIELDS}
EVAL_FILES = ["eval_set.jsonl", "heldout_set.jsonl"] + [f"heldout{i}_set.jsonl" for i in range(2, 10)]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(name):
    return [json.loads(line) for line in (ROOT / "data" / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_data():
    expected = json.loads((ROOT / "dataset-lock.json").read_text())
    train_texts = set()
    labels = {}
    evaluation_texts = set()
    for name, info in expected.items():
        if sha256(ROOT / "data" / name) != info["sha256"]:
            raise ValueError(f"Dataset changed: {name}. Do not rebuild the frozen comparison data.")
        rows = read_rows(name)
        if len(rows) != info["count"]:
            raise ValueError(f"Row count changed: {name}")
        seen = set()
        for row in rows:
            text = " ".join(row["text"].lower().split())
            exp = row["expected"]
            if set(exp) != set(FIELDS) or any(exp[f] not in ENUMS[f] for f in FIELDS):
                raise ValueError(f"Invalid V3 label in {name}: {row}")
            if exp["until_region"] != "none" and exp["stay_region"] == "none":
                raise ValueError(f"until_region without stay_region in {name}")
            if text in labels and labels[text] != exp:
                raise ValueError(f"Conflicting labels: {text}")
            labels[text] = exp
            if name != "train_set.jsonl" and text in seen:
                raise ValueError(f"Duplicate evaluation command in {name}: {text}")
            seen.add(text)
        if name == "train_set.jsonl":
            train_texts = seen  # Oversampling is intentional; do not deduplicate.
        else:
            evaluation_texts.update(seen)
    if train_texts & evaluation_texts:
        raise ValueError("Training/evaluation command overlap")
    source = json.loads((ROOT / "source-v3.json").read_text())
    if sha256(ROOT / "core" / "schema.py") != source["schema_sha256"]:
        raise ValueError("Schema differs from the locked V3 source")
    return expected


def eval_rows(selection="all"):
    names = EVAL_FILES if selection == "all" else [selection]
    if any(name not in EVAL_FILES for name in names):
        raise ValueError("Select an evaluation file, not training data")
    return [{**r, "dataset": name, "row_id": i + 1} for name in names
            for i, r in enumerate(read_rows(name))]


def messages(key, text):
    if key == "functiongemma":
        return build_messages(text)
    # Same instructional content; LFM uses system rather than Gemma's developer role.
    return [{"role": "system", "content": build_messages(text)[0]["content"]},
            {"role": "user", "content": text}]


def prompt(key, formatter, text):
    tools = [FUNCTION_SCHEMA] if key == "functiongemma" else [FUNCTION_SCHEMA["function"]]
    kwargs = {"tools": tools, "tokenize": False, "add_generation_prompt": True}
    return formatter.apply_chat_template(messages(key, text), **kwargs)


def completion(key, expected):
    if key == "functiongemma":
        parts = ",".join(f"{f}:<escape>{expected[f]}<escape>" for f in FIELDS)
        return f"<start_function_call>call:set_leader_intent{{{parts}}}<end_function_call><end_of_turn>\n"
    if key in LFM_MODELS:
        # Native LFM tool-call format, with identical labels and field order.
        parts = ", ".join(f"{f}={expected[f]!r}" for f in FIELDS)
        return f"<|tool_call_start|>[set_leader_intent({parts})]<|tool_call_end|><|im_end|>\n"
    if key in JSON_TOOL_MODELS:
        payload = {"name": "set_leader_intent",
                   "arguments": {field: expected[field] for field in FIELDS}}
        end = "<|end_of_text|>"
        return f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>{end}\n"
    raise KeyError(f"Unsupported model: {key}")


def _json_without_duplicate_keys(text):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=object_pairs)


def parse(key, raw):
    """Parse one complete native call; reject wrong tools, duplicates and extra calls.

    Missing fields remain None for per-field diagnostics. A malformed wrapper yields
    no parsed fields. No semantic repair or label-dependent fallback is performed.
    """
    empty = dict.fromkeys(FIELDS)
    body = raw.strip()
    for token in stop_tokens(key):
        if body.endswith(token):
            body = body[:-len(token)].rstrip()
    try:
        if key == "functiongemma":
            match = re.fullmatch(r"<start_function_call>call:set_leader_intent\{(.*)\}<end_function_call>", body, re.DOTALL)
            if not match:
                return empty
            pairs = re.findall(r"(\w+):<escape>([^<>]*)<escape>", match[1])
            if ",".join(f"{k}:<escape>{v}<escape>" for k,v in pairs) != match[1]:
                return empty
            values = {}
            for k,v in pairs:
                if k not in FIELDS or k in values:
                    return empty
                values[k] = v
        elif key == "granite350":
            match = re.fullmatch(r"<tool_call>\s*(.*?)\s*</tool_call>", body, re.DOTALL)
            if not match:
                return empty
            obj = _json_without_duplicate_keys(match[1])
            if not isinstance(obj, dict) or set(obj) != {"name", "arguments"} or obj["name"] != "set_leader_intent":
                return empty
            values = obj["arguments"]
            if not isinstance(values, dict) or not set(values).issubset(FIELDS):
                return empty
        elif key == "lfm350":
            match = re.fullmatch(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", body, re.DOTALL)
            if not match:
                return empty
            node = ast.parse(match[1].strip(), mode="eval").body
            if not isinstance(node, ast.List) or len(node.elts) != 1:
                return empty
            call = node.elts[0]
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "set_leader_intent" or call.args:
                return empty
            values = {}
            for item in call.keywords:
                if item.arg not in FIELDS or item.arg in values:
                    return empty
                if not isinstance(item.value, ast.Constant) or not isinstance(item.value.value, str):
                    return empty
                values[item.arg] = item.value.value
        else:
            raise KeyError(key)
        if not all(isinstance(v, str) for v in values.values()):
            return empty
        return {f: values.get(f) for f in FIELDS}
    except (SyntaxError, ValueError, TypeError):
        return empty


def masked_example(tokenizer, prompt_text, target):
    ids = tokenizer(prompt_text + target, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    # Fail rather than silently mask the wrong span if a tokenizer merges across the boundary.
    assert ids[:len(prompt_ids)] == prompt_ids, "Prompt/completion token boundary mismatch"
    assert len(ids) > len(prompt_ids)
    return {"input_ids": ids, "labels": [-100] * len(prompt_ids) + ids[len(prompt_ids):]}


def stop_tokens(key):
    if key == "functiongemma":
        return ["<end_of_turn>"]
    if key == "granite350":
        return ["<|end_of_text|>"]
    return ["<|im_end|>"]


def call_tokens(key):
    if key in LFM_MODELS:
        return ["<|tool_call_start|>", "<|tool_call_end|>"]
    if key in JSON_TOOL_MODELS:
        return ["<tool_call>", "</tool_call>"]
    return []


def summarize(examples):
    n = len(examples)
    exact = sum(e["exact_match"] for e in examples)
    result = {"n_examples": n, "n_correct": exact, "exact_match_accuracy": exact / n,
              "per_field_accuracy": {f: sum(e["parsed"][f] == e["expected"][f] for e in examples) / n for f in FIELDS}}
    values = sorted(e["latency_s"] for e in examples)
    result["latency_s"] = {"mean": statistics.mean(values), "median": statistics.median(values),
                           "p95": values[max(0, math.ceil(.95 * n) - 1)], "min": values[0], "max": values[-1]}
    result["token_limit_count"] = sum(bool(e.get("hit_token_limit") or e.get("stopped_limit")) for e in examples)
    result["invalid_output_count"] = sum(not e["schema_valid"] for e in examples)
    result["cross_field_violation_count"] = sum(not e["cross_field_valid"] for e in examples)
    for group_key in ["dataset", "action"]:
        groups = {}
        for e in examples:
            group = e["expected"]["action"] if group_key == "action" else e.get("dataset", "unknown")
            hits, total = groups.get(group, (0, 0))
            groups[group] = (hits + int(e["exact_match"]), total + 1)
        result["by_" + group_key] = {k: {"correct": h, "n": n, "exact_match_accuracy": h/n} for k,(h,n) in groups.items()}
    result["subsets"] = {}
    for group in sorted({e.get("set", "unmarked") for e in examples}):
        part = [e for e in examples if e.get("set", "unmarked") == group]
        result["subsets"][group] = {"n": len(part), "correct": sum(e["exact_match"] for e in part),
                                    "exact_match_accuracy": sum(e["exact_match"] for e in part) / len(part)}
    return result


def scored_row(row, raw, seconds, key, **extras):
    parsed = parse(key, raw)
    return {**row, "raw_output": raw, "parsed": parsed, "latency_s": seconds,
            "exact_match": all(parsed[f] == row["expected"][f] for f in FIELDS),
            "schema_valid": all(parsed[f] in ENUMS[f] for f in FIELDS),
            "cross_field_valid": not (parsed["until_region"] not in (None, "none") and parsed["stay_region"] in (None, "none")), **extras}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
