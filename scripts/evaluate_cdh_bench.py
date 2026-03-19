import http.client
import base64
import hashlib
import json
import os
import time
import ssl
import argparse
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from cdh_bench_loader import CDHBenchLoader


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_subcategory(subcategory: str) -> str:
    return (subcategory or "").replace(" ", "_").replace("/", "_")


def _normalize_pair_id(pair_id: str) -> str:
    return (pair_id or "").replace(" ", "_")


def _image_path(images_root: str, subcategory: str, pair_id: str, side: str) -> str:
    sub_dir = _normalize_subcategory(subcategory)
    p_dir = _normalize_pair_id(pair_id)
    filename = "counterfactual.png" if side == "counterfactual" else "commonsense.png"
    return str(Path(images_root) / sub_dir / p_dir / filename)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _safe_slug(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "model"
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)[:120]


def _hash_dict(d: Dict[str, Any]) -> str:
    raw = json.dumps(d, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def _strip_thinking(text: str) -> str:
    """如果文本包含 <think>...</think>，则返回 </think> 之后的内容。"""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def _extract_first_letter(text: str) -> Optional[str]:
    text = _strip_thinking(text)
    t = text.strip().upper()
    patterns = [
        r'^([A-D])(?:\.|\)|$|\s)',
        r'(?:ANSWER|答案)(?:IS|是|为)?\s*([A-D])',
        r'\s([A-D])(?:\.|\)|$|\s)',
    ]
    for p in patterns:
        m = re.search(p, t)
        if m:
            return m.group(1)
    
    m = re.search(r'\b([A-D])\b', t)
    if m:
        return m.group(1)
        
    return None


def _extract_first_int(text: str) -> Optional[int]:
    text = _strip_thinking(text)
    m = re.search(r'\d+', text)
    if m:
        try:
            return int(m.group())
        except:
            pass
    return None


def _extract_yes_no(text: str) -> Optional[str]:
    text = _strip_thinking(text)
    t = _normalize_text(text)
    if not t:
        return None
    
    words = t.split()
    if words:
        first = words[0]
        if first in ("yes", "y", "true", "是", "对"):
            return "yes"
        if first in ("no", "n", "false", "否", "不", "不是", "错"):
            return "no"
    
    if "yes" in words or "是" in t or "对" in t:
        return "yes"
    if "no" in words or "否" in t or "不" in t:
        return "no"
        
    return None


def _normalize_text(text: str) -> str:
    text = _strip_thinking(text)
    t = (text or "").strip().lower()
    t = "".join(ch for ch in t if ch.isalnum() or ch.isspace())
    t = " ".join(t.split())
    return t


def _score_direct_qa(pred: str, gt: str) -> bool:
    gt = (gt or "").strip()
    if gt == "":
        return False
    gt_norm = _normalize_text(gt)
    pred_norm = _normalize_text(pred)
    if gt_norm in ("yes", "no"):
        p = _extract_yes_no(pred)
        return p == gt_norm
    if gt.isdigit():
        p = _extract_first_int(pred)
        return p is not None and str(p) == gt
    if gt_norm == pred_norm:
        return True
    return gt_norm in pred_norm


def _score_multiple_choice(pred: str, gt_letter: str) -> bool:
    gt_letter = (gt_letter or "").strip().upper()
    if gt_letter not in ("A", "B", "C", "D"):
        return False
    p = _extract_first_letter(pred)
    return p == gt_letter


def _task_fields(task: str) -> Tuple[str, str]:
    if task == "qa":
        return "direct_qa", "question"
    if task == "mc":
        return "multiple_choice", "question"
    raise ValueError(f"unknown task: {task}")


def _get_gt(item: Dict[str, Any], task: str, side: str) -> str:
    if task == "qa":
        return (item.get("direct_qa") or {}).get(f"{side}_gt") or ""
    if task == "mc":
        return (item.get("multiple_choice") or {}).get(f"{side}_gt") or ""
    raise ValueError(f"unknown task: {task}")


def _get_question(item: Dict[str, Any], task: str) -> str:
    if task == "qa":
        return (item.get("direct_qa") or {}).get("question") or ""
    if task == "mc":
        return (item.get("multiple_choice") or {}).get("question") or ""
    raise ValueError(f"unknown task: {task}")


def _get_options(item: Dict[str, Any]) -> List[str]:
    opts = (item.get("multiple_choice") or {}).get("options") or []
    if isinstance(opts, list):
        return [str(x) for x in opts]
    return []


def _build_user_text(task: str, item: Dict[str, Any]) -> str:
    q = _get_question(item, task)
    if task == "mc":
        opts = _get_options(item)
        opts_text = "\n".join(str(o) for o in opts)
        return f"{q}\n{opts_text}\nAnswer with a single letter (A, B, C, or D)."
    if task == "qa":
        return f"{q}\nAnswer with yes or no."
    raise ValueError(f"unknown task: {task}")


def _b64_data_url(path: str) -> str:
    from PIL import Image
    import io
    img = Image.open(path).convert("RGB")
    img = img.resize((512, 512), Image.Resampling.LANCZOS)
    
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backend: str
    model: str
    base_url: str
    api_key: Optional[str]
    temperature: float
    max_tokens: int
    models_root: str


def _parse_models_arg(models_arg: str) -> List[Dict[str, Any]]:
    p = Path(models_arg)
    if p.exists() and p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(models_arg)


def _load_model_specs(models_arg: str) -> List[ModelSpec]:
    raw = _parse_models_arg(models_arg)
    if not isinstance(raw, list):
        raise ValueError("--models must be a JSON list or a path to a JSON file containing a list")
    specs: List[ModelSpec] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or entry.get("model") or "model")
        backend = str(entry.get("backend") or "").lower().strip()
        model = str(entry.get("model") or "")
        base_url = str(entry.get("base_url") or "").rstrip("/")
        api_key = entry.get("api_key")
        api_key_env = entry.get("api_key_env")
        if api_key is None and api_key_env:
            api_key = os.environ.get(str(api_key_env))
        temperature = float(entry.get("temperature", 0.0))
        max_tokens = int(entry.get("max_tokens", 128))
        models_root = str(entry.get("models_root") or os.environ.get("CDH_MODELS_ROOT") or "/home/cks/cdh-ben/models")
        if backend not in ("api", "vllm"):
            raise ValueError(f"unknown backend for model {name}: {backend}")
        if backend == "api" and not base_url:
            raise ValueError(f"missing base_url for model {name}")
        if not model:
            raise ValueError(f"missing model for model {name}")
        specs.append(
            ModelSpec(
                name=name,
                backend=backend,
                model=model,
                base_url=base_url,
                api_key=str(api_key) if api_key is not None else None,
                temperature=temperature,
                max_tokens=max_tokens,
                models_root=models_root,
            )
        )
    return specs


def _http_post_json(url_str: str, payload: Dict[str, Any], api_key: Optional[str], timeout_s: int) -> Dict[str, Any]:
    from urllib.parse import urlparse
    parsed = urlparse(url_str)
    host = parsed.hostname
    port = parsed.port
    path = parsed.path
    if parsed.query:
        path += "?" + parsed.query
    
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    
    payload_json = json.dumps(payload).encode("utf-8")
    
    if parsed.scheme == "https":
        conn = http.client.HTTPSConnection(host, port if port else 443, timeout=timeout_s, context=ssl._create_unverified_context())
    else:
        conn = http.client.HTTPConnection(host, port if port else 80, timeout=timeout_s)
    
    try:
        conn.request("POST", path, body=payload_json, headers=headers)
        res = conn.getresponse()
        data = res.read().decode("utf-8")
        if res.status >= 400:
            raise RuntimeError(f"HTTP {res.status}: {data}")
        return json.loads(data)
    finally:
        conn.close()


def _call_openai_compat_chat(
    spec: ModelSpec,
    user_text: str,
    image_path: str,
    timeout_s: int,
) -> Tuple[str, Dict[str, Any]]:
    base_url = spec.base_url.rstrip("/")
    if base_url.endswith("/v1/chat/completions"):
        url = base_url
    elif base_url.endswith("/v1"):
        url = f"{base_url}/chat/completions"
    else:
        url = f"{base_url}/v1/chat/completions"
    
    payload = {
        "model": spec.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": _b64_data_url(image_path)}},
                ],
            }
        ],
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
        "stream": False
    }
    raw = _http_post_json(url, payload, spec.api_key, timeout_s=timeout_s)
    text = ""
    try:
        text = raw["choices"][0]["message"]["content"]
    except Exception:
        text = json.dumps(raw, ensure_ascii=False)
    return str(text), raw


_LOCAL_QWEN3_CACHE: Dict[str, Any] = {}


def _call_local_qwen3_vl_chat(
    spec: ModelSpec,
    user_text: str,
    image_path: str,
) -> Tuple[str, Dict[str, Any]]:
    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
    except Exception as e:
        raise RuntimeError(
            "local vllm backend requires torch+Pillow+transformers+bitsandbytes in the current Python environment"
        ) from e

    cache_key = f"{spec.models_root}::{spec.model}"
    bundle = _LOCAL_QWEN3_CACHE.get(cache_key)
    if bundle is None:
        model_path = Path(spec.models_root) / spec.model
        load_id = str(model_path) if model_path.exists() else spec.model
        
        quant_config = None
        if "32B" in spec.model or "235B" in spec.model:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            load_id, 
            dtype="auto", 
            device_map="auto",
            quantization_config=quant_config
        )
        processor = AutoProcessor.from_pretrained(load_id)
        bundle = (model, processor)
        _LOCAL_QWEN3_CACHE[cache_key] = bundle

    model, processor = bundle
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(spec.max_tokens)}
    if float(spec.temperature) and float(spec.temperature) > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(spec.temperature)
    else:
        gen_kwargs["do_sample"] = False

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        out_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return str(out_text), {"backend": "local_qwen3_vl", "model": spec.model}

    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)]
    out_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return str(out_text), {"backend": "local_qwen3_vl", "model": spec.model}


def _call_model(
    spec: ModelSpec,
    user_text: str,
    image_path: str,
    timeout_s: int,
) -> Tuple[str, Dict[str, Any]]:
    if spec.backend == "api":
        return _call_openai_compat_chat(spec, user_text=user_text, image_path=image_path, timeout_s=timeout_s)
    if spec.backend == "vllm":
        return _call_local_qwen3_vl_chat(spec, user_text=user_text, image_path=image_path)
    raise ValueError(f"unknown backend: {spec.backend}")


def _score(task: str, pred: str, gt: str) -> bool:
    if task == "qa":
        return _score_direct_qa(pred, gt)
    if task == "mc":
        return _score_multiple_choice(pred, gt)
    raise ValueError(f"unknown task: {task}")


def _collect_existing_keys(results_jsonl: str) -> set[Tuple[str, str, str]]:
    existing = set()
    for rec in _read_jsonl(results_jsonl):
        if rec.get("status") != "ok":
            continue
        pid = str(rec.get("pair_id") or "")
        task = str(rec.get("task") or "")
        side = str(rec.get("side") or "")
        if pid and task and side:
            existing.add((pid, task, side))
    return existing


def _aggregate_metrics(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    correct = 0
    cf_total = 0
    cf_correct = 0
    cs_total = 0
    cs_correct = 0
    cf_errors = 0
    cf_commonsense_errors = 0
    for r in records:
        if r.get("status") != "ok":
            continue
        total += 1
        if r.get("correct") is True:
            correct += 1
        side = r.get("side")
        if side == "counterfactual":
            cf_total += 1
            if r.get("correct") is True:
                cf_correct += 1
            else:
                cf_errors += 1
                if r.get("commonsense_error") is True:
                    cf_commonsense_errors += 1
        elif side == "commonsense":
            cs_total += 1
            if r.get("correct") is True:
                cs_correct += 1

    cf_acc = (cf_correct / cf_total) if cf_total else None
    cs_acc = (cs_correct / cs_total) if cs_total else None
    gap = (cs_acc - cf_acc) if (cs_acc is not None and cf_acc is not None) else None
    ccr = (cf_commonsense_errors / cf_errors) if cf_errors else None
    rpd = ((cs_acc - cf_acc) / cs_acc) if (cs_acc is not None and cf_acc is not None and cs_acc not in (0, None)) else None
    return {
        "n_total": total,
        "n_cf": cf_total,
        "n_cs": cs_total,
        "CF_Acc": cf_acc,
        "CS_Acc": cs_acc,
        "Gap": gap,
        "CCR": ccr,
        "RPD": rpd,
    }


def _build_summary(all_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_records:
        by_task.setdefault(str(r.get("task")), []).append(r)

    out: Dict[str, Any] = {"overall": {}, "by_category": {}, "by_subcategory": {}}

    for task, recs in by_task.items():
        out["overall"][task] = _aggregate_metrics(recs)

        cat_map: Dict[str, List[Dict[str, Any]]] = {}
        sub_map: Dict[str, List[Dict[str, Any]]] = {}
        for rr in recs:
            cat = str(rr.get("category") or "Unknown")
            sub = str(rr.get("subcategory") or "Unknown")
            cat_map.setdefault(cat, []).append(rr)
            sub_key = f"{cat} / {sub}"
            sub_map.setdefault(sub_key, []).append(rr)

        out["by_category"][task] = {k: _aggregate_metrics(v) for k, v in sorted(cat_map.items(), key=lambda x: x[0])}
        out["by_subcategory"][task] = {k: _aggregate_metrics(v) for k, v in sorted(sub_map.items(), key=lambda x: x[0])}

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="/home/cks/cdh-ben/CDH-Bench.jsonl")
    ap.add_argument("--images-root", default="/home/cks/cdh-ben/images")
    ap.add_argument("--output-dir", default="/home/cks/cdh-ben/result")
    ap.add_argument("--models", default="")
    ap.add_argument("--tasks", default="qa,mc")
    ap.add_argument("--timeout-s", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=1, help="Number of parallel API requests")
    ap.add_argument("--retry", type=int, default=3, help="Number of retries for failed requests")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tasks:
        if t not in ("qa", "mc"):
            raise SystemExit(f"invalid task: {t}")

    loader = CDHBenchLoader(args.jsonl)
    items = loader.data
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    models_arg = str(args.models or "").strip()
    if not models_arg:
        base_url = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
        temperature = float(os.environ.get("CDH_EVAL_TEMPERATURE", "0.0"))
        max_tokens = int(os.environ.get("CDH_EVAL_MAX_TOKENS", "4096"))
        specs = [
            ModelSpec(
                name="Qwen3-VL-2B-Instruct", backend="vllm", model="Qwen3-VL-2B-Instruct",
                base_url=base_url, temperature=temperature, max_tokens=max_tokens, models_root="/home/cks/cdh-ben/models"
            )
        ]
    else:
        specs = _load_model_specs(models_arg)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_lock = threading.Lock()

    # 1. Collect all evaluation tasks from all models
    global_eval_tasks = []
    for spec in specs:
        config_hash = _hash_dict({
            "model": spec.model,
            "backend": spec.backend,
            "base_url": spec.base_url,
            "temp": spec.temperature,
            "max_tokens": spec.max_tokens,
        })
        model_dir = output_dir / _safe_slug(spec.name)
        model_dir.mkdir(parents=True, exist_ok=True)
        results_path = str(model_dir / "results.jsonl")
        
        existing_keys = _collect_existing_keys(results_path)
        
        for item in items:
            pair_id = str(item.get("pair_id") or "")
            category = str(item.get("category") or "")
            subcategory = str(item.get("subcategory") or "")
            for t_type in tasks:
                for side in ["commonsense", "counterfactual"]:
                    if (pair_id, t_type, side) in existing_keys:
                        continue
                    global_eval_tasks.append({
                        "spec": spec,
                        "item": item,
                        "task": t_type,
                        "side": side,
                        "pair_id": pair_id,
                        "category": category,
                        "subcategory": subcategory,
                        "results_path": results_path,
                        "config_hash": config_hash,
                        "model_dir": model_dir
                    })

    def process_task(task_ctx: Dict[str, Any]) -> Dict[str, Any]:
        spec = task_ctx["spec"]
        item = task_ctx["item"]
        task = task_ctx["task"]
        side = task_ctx["side"]
        pair_id = task_ctx["pair_id"]
        category = task_ctx["category"]
        subcategory = task_ctx["subcategory"]
        results_path = task_ctx["results_path"]
        config_hash = task_ctx["config_hash"]

        img_path = _image_path(args.images_root, subcategory, pair_id, side)
        if not os.path.exists(img_path):
            rec = {
                "ts": _utc_now_iso(), "run": config_hash, "model_name": spec.name,
                "backend": spec.backend, "model": spec.model, "pair_id": pair_id,
                "category": category, "subcategory": subcategory, "task": task,
                "side": side, "image_path": img_path, "status": "missing_image",
            }
            with file_lock:
                _append_jsonl(results_path, rec)
            return rec

        user_text = _build_user_text(task, item)
        gt = _get_gt(item, task, side)
        cs_gt = _get_gt(item, task, "commonsense")
        cf_gt = _get_gt(item, task, "counterfactual")

        last_err = ""
        dt_ms = 0
        status = "ok"
        pred_text = ""
        raw_resp = {}
        for attempt in range(args.retry + 1):
            t0 = time.time()
            status = "ok"
            try:
                pred_text, raw_resp = _call_model(spec, user_text=user_text, image_path=img_path, timeout_s=args.timeout_s)
                dt_ms = int((time.time() - t0) * 1000)
                break
            except Exception as e:
                status = "error"
                pred_text = str(e)
                dt_ms = int((time.time() - t0) * 1000)
                last_err = pred_text
                if attempt < args.retry:
                    time.sleep(2 ** attempt)
                    continue
        
        correct = False
        commonsense_error = False
        if status == "ok":
            correct = _score(task, pred_text, gt)
            if side == "counterfactual" and (not correct):
                commonsense_error = _score(task, pred_text, cs_gt)

        rec = {
            "ts": _utc_now_iso(), "run": config_hash, "model_name": spec.name,
            "backend": spec.backend, "model": spec.model, "pair_id": pair_id,
            "category": category, "subcategory": subcategory, "task": task,
            "side": side, "image_path": img_path, "status": status,
            "latency_ms": dt_ms, "question": _get_question(item, task),
            "prompt": item.get(f"{side}_prompt"), "gt": gt, "cf_gt": cf_gt,
            "cs_gt": cs_gt, "pred": pred_text,
            "correct": bool(correct) if status == "ok" else None,
            "commonsense_error": bool(commonsense_error) if (status == "ok" and side == "counterfactual") else None,
            "raw": raw_resp if status == "ok" else None,
        }
        with file_lock:
            _append_jsonl(results_path, rec)
        return rec

    # 2. Split tasks into API (parallel) and local (sequential)
    api_tasks = [t for t in global_eval_tasks if t["spec"].backend == "api"]
    local_tasks = [t for t in global_eval_tasks if t["spec"].backend == "vllm"]

    # 3. Run API tasks in parallel
    if api_tasks:
        print(f"Starting parallel API evaluation for {len(api_tasks)} tasks across {len(specs)} models...")
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = [executor.submit(process_task, t) for t in api_tasks]
            for future in tqdm(as_completed(futures), total=len(api_tasks), desc="API Eval Progress"):
                future.result()

    # 4. Run local tasks sequentially
    if local_tasks:
        print(f"Starting sequential local evaluation for {len(local_tasks)} tasks...")
        for t in tqdm(local_tasks, desc="Local Eval Progress"):
            process_task(t)

    # 5. Build summary for each model
    for spec in specs:
        model_dir = output_dir / _safe_slug(spec.name)
        results_path = model_dir / "results.jsonl"
        if results_path.exists():
            records = _read_jsonl(str(results_path))
            summary = _build_summary(records)
            (model_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
