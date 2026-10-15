"""
Handles the model-specific details needed to turn a trained model into a usable GGUF
file. It checks the training record and converter version, prepares compatible model
settings, and creates the Q8 file with a record of its source. Step 04 calls this
automatically.
"""
# Previously: gguf_workflow.py
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from model_common import ROOT, MODELS, sha256, verify_data, write_json

MODEL_KEYS = list(MODELS)
from artifacts import LLAMA_DIR, CONVERTER_PYTHON, binary
CONVERTER = LLAMA_DIR / "convert_hf_to_gguf.py"
SERVER = binary('llama-server')
EXPECTED_TAG = "b11026"
DECIMAL_GB = 1_000_000_000


def run(command, **kwargs):
    print("Running:", " ".join(str(x) for x in command), flush=True)
    return subprocess.run([str(x) for x in command], check=True, **kwargs)


def completed_run(key):
    path = ROOT / "checkpoints" / f"{key}-lora" / "run.json"
    if not path.exists():
        raise SystemExit(f"Missing completed training run: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not value.get("completed") or value.get("smoke_only"):
        raise SystemExit(f"Training run is not a completed full run: {path}")
    if value["datasets"] != verify_data():
        raise SystemExit(f"Dataset lock differs from training run: {path}")
    from model_workflow import source_path
    if value["source"] != json.loads((source_path(key) / "source.json").read_text()):
        raise ValueError("Base revision differs from training")
    if value["adapter_sha256"] != sha256(path.parent / "adapter_model.safetensors"):
        raise ValueError("Adapter weights differ from training manifest")
    return value


def prepare_tokenizer_for_converter(model, merged):
    """Make Transformers 5 tokenizer metadata readable by the pinned converter.

    Transformers 5 can save the implementation detail ``TokenizersBackend`` as
    tokenizer_class.  The converter intentionally uses Transformers 4.57.6,
    where that class name does not exist.  Restore the class from the untouched
    downloaded model when possible, otherwise use the equivalent generic fast
    tokenizer class.  This only edits the reconstructable merged export.
    """
    merged_config_path = merged / "tokenizer_config.json"
    source_config_path = ROOT / "models" / model / "tokenizer_config.json"
    if not merged_config_path.exists() or not source_config_path.exists():
        raise SystemExit(f"Missing tokenizer_config.json for converter preparation: {merged}")

    merged_config = json.loads(merged_config_path.read_text(encoding="utf-8"))
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    saved_class = merged_config.get("tokenizer_class")
    source_class = source_config.get("tokenizer_class")
    converter_class = saved_class
    changes = []

    if saved_class == "TokenizersBackend":
        converter_class = source_class
        if not converter_class or converter_class == "TokenizersBackend":
            converter_class = "PreTrainedTokenizerFast"
        merged_config["tokenizer_class"] = converter_class
        changes.append({"field": "tokenizer_class", "from": saved_class, "to": converter_class})

    # Transformers 4 treats this newer field as a named-token mapping.  A list
    # makes tokenizer construction fail before conversion; tokenizer.json still
    # retains all token IDs and added-token definitions.
    if isinstance(merged_config.get("extra_special_tokens"), list):
        changes.append({"field": "extra_special_tokens", "action": "removed_transformers_5_only_list"})
        del merged_config["extra_special_tokens"]

    if changes:
        write_json(merged_config_path, merged_config)

    return {
        "source_tokenizer_class": source_class,
        "saved_tokenizer_class": saved_class,
        "converter_tokenizer_class": converter_class,
        "changes_applied_this_run": changes,
    }


def prepare_model_config_for_converter(model, merged):
    """Normalize known Transformers 5 architecture metadata for llama.cpp."""
    config_path = merged / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    changes = []

    layer_types = config.get("layer_types")
    all_attention = (
        isinstance(layer_types, list)
        and layer_types
        and all(value in {"attention", "full_attention"} for value in layer_types)
    )
    if model == "granite350" and all_attention:
        granite_source = LLAMA_DIR / "conversion" / "granite.py"
        marker = "AVC_GRANITE_ALL_ATTENTION_WRITER_FIX"
        if marker not in granite_source.read_text(encoding="utf-8"):
            raise SystemExit(
                "Pinned llama.cpp is missing the recorded Granite all-attention patch. "
                "Apply patches/llama-b11026-granite-all-attention.patch as documented in docs/SETUP.md."
            )

    if isinstance(layer_types, list) and "full_attention" in layer_types:
        config["layer_types"] = [
            "attention" if value == "full_attention" else value for value in layer_types
        ]
        changes.append({
            "field": "layer_types",
            "from_value": "full_attention",
            "to_value": "attention",
        })

    if changes:
        write_json(config_path, config)
    return {"changes_applied_this_run": changes}


def llama_version():
    try:
        tag = subprocess.check_output(
            ["git", "-C", str(LLAMA_DIR), "describe", "--tags", "--exact-match"], text=True
        ).strip()
        commit = subprocess.check_output(
            ["git", "-C", str(LLAMA_DIR), "rev-parse", "HEAD"], text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise SystemExit("llama.cpp is missing or is not at a tagged revision. Follow docs/SETUP.md.")
    if tag != EXPECTED_TAG or commit != 'b49650adb31f2e49a0d76113aeb1792134fd8413':
        raise SystemExit(f"Expected llama.cpp {EXPECTED_TAG}; found {tag}. Keep one pinned build for every model.")
    diff = subprocess.check_output(["git", "-C", str(LLAMA_DIR), "diff", "--binary", "HEAD"], text=False)
    return {
        "tag": tag,
        "commit": commit,
        "local_patch_sha256": hashlib.sha256(diff).hexdigest() if diff else None,
    }


def require_tools(check_architectures=True):
    missing = [p for p in [CONVERTER, CONVERTER_PYTHON, SERVER] if not p.exists()]
    if missing:
        raise SystemExit("Missing GGUF setup files:\n" + "\n".join(str(p) for p in missing) +
                         "\nComplete the setup section in docs/SETUP.md.")
    version = llama_version()
    if check_architectures:
        result = subprocess.run([str(CONVERTER_PYTHON), str(CONVERTER), "--print-supported-models"],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        for key in MODEL_KEYS:
            if not (ROOT / 'models' / key / 'config.json').exists():
                continue
            config = json.loads((ROOT / "models" / key / "config.json").read_text(encoding="utf-8"))
            architecture = config["architectures"][0]
            if architecture not in result.stdout:
                raise SystemExit(f"Pinned converter does not list {architecture} for {key}.")
    return version


def check(_args):
    verify_data()
    version = require_tools()
    usage = shutil.disk_usage(ROOT)
    print(json.dumps({
        "status": "toolchain ready; training must finish before conversion",
        "models": MODEL_KEYS,
        "llama_cpp": version,
        "free_disk_bytes": usage.free,
        "free_disk_gib": round(usage.free / (1024 ** 3), 2),
        "conversion": "merged FP32 LoRA checkpoint -> direct Q8_0 GGUF",
        "size_limit_bytes": DECIMAL_GB,
    }, indent=2))


def convert(args):
    verify_data()
    training = completed_run(args.model)
    version = require_tools()
    merged = ROOT / "merged" / args.model
    prompts = ROOT / "prompts" / f"{args.model}.json"
    output = ROOT / "gguf" / f"{args.model}-q8_0.gguf"
    manifest = ROOT / "gguf" / f"{args.model}-q8_0.json"
    if output.exists() or manifest.exists():
        raise SystemExit(f"Q8_0 output already exists for {args.model}. Preserve and benchmark it: {output}")
    # A successful --cleanup-merged deliberately preserves the prompt bundle,
    # so prompts-without-merged is the normal state for a later reconversion.
    # Merged-without-prompts is the genuinely incomplete export.
    if merged.exists() and not prompts.exists():
        raise SystemExit("Partial export detected. Preserve it and inspect before continuing: " + str(merged))
    if not merged.exists():
        run([sys.executable, ROOT / "core" / "model_workflow.py", "export", "--model", args.model,
             "--device", "cpu", "--threads", str(args.threads)])
    if json.loads((merged / "training-run.json").read_text()) != training:
        raise ValueError("Merged checkpoint belongs to a different training run")
    exported = json.loads((merged / "export-metadata.json").read_text())
    for filename, digest in exported["weight_files"].items():
        if sha256(merged / filename) != digest:
            raise ValueError("Merged weights changed after export")
    model_config_compatibility = prepare_model_config_for_converter(args.model, merged)
    tokenizer_compatibility = prepare_tokenizer_for_converter(args.model, merged)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".gguf.partial")
    if temporary.exists():
        raise SystemExit(f"Previous incomplete conversion: {temporary}. Inspect/remove it before retrying.")
    run([CONVERTER_PYTHON, CONVERTER, merged, "--outfile", temporary, "--outtype", "q8_0"])
    temporary.rename(output)
    with output.open("rb") as file:
        header = file.read(4)
    if output.stat().st_size < 4 or header != b"GGUF":
        raise SystemExit(f"Converter did not create a valid GGUF header: {output}")
    info = {
        "model_key": args.model,
        "repo_id": MODELS[args.model],
        "source": training["source"],
        "datasets": training["datasets"],
        "training_run": str((ROOT / "checkpoints" / f"{args.model}-lora" / "run.json").relative_to(ROOT)),
        "prompt_bundle": str(prompts.relative_to(ROOT)),
        "prompt_sha256": sha256(prompts),
        "adapter_sha256": training["adapter_sha256"],
        "training_manifest_sha256": sha256(ROOT / "checkpoints" / f"{args.model}-lora" / "run.json"),
        "quantization": "Q8_0",
        "conversion_path": "merged FP32 PEFT model converted directly by convert_hf_to_gguf.py",
        "model_config_converter_compatibility": model_config_compatibility,
        "tokenizer_converter_compatibility": tokenizer_compatibility,
        "llama_cpp": version,
        "gguf_file": output.name,
        "gguf_bytes": output.stat().st_size,
        "gguf_sha256": sha256(output),
        "decimal_1gb_limit": DECIMAL_GB,
        "passes_decimal_1gb_limit": output.stat().st_size <= DECIMAL_GB,
    }
    write_json(manifest, info)
    print(json.dumps(info, indent=2))
    print("Created:", output)
    print("Next: benchmark every V3 evaluation example before judging the converted model.")
