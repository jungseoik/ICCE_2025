#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, os, re
from typing import Any, Dict, List, Optional

# ====== CONFIG (edit here) ======
JSON_ROOT   = "HAWK_bench_json"  # 폴더 전체 스캔
BACKUP_EXT  = ".bak"             # 백업 확장자
INDENT      = 2                  # pretty json indent
# =================================

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

def read_text(p: str) -> str:
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def write_json(p: str, d: Dict[str, Any]) -> None:
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=INDENT)
        f.write("\n")

def backup_once(p: str) -> str:
    b = p + BACKUP_EXT
    if not os.path.exists(b):
        with open(p, "rb") as s, open(b, "wb") as t:
            t.write(s.read())
    return b

def try_load(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        return None

def strip_code_fence(s: str) -> str:
    m = FENCE_RE.search(s or "")
    if m:
        return m.group(1).strip()
    return (s or "").strip()

def extract_key_string(key: str, s: str) -> Optional[str]:
    pat = re.compile(rf'"{re.escape(key)}"\s*:\s*"(.+?)"', re.DOTALL)
    m = pat.search(s)
    return m.group(1).strip() if m else None

def extract_options_block(s: str) -> List[str]:
    opts: List[str] = []
    lines = s.splitlines()
    in_opts = False
    for ln in lines:
        if not in_opts:
            if re.search(r'"options"\s*:\s*\[', ln):
                in_opts = True
            continue
        if "]" in ln:
            items = re.findall(r'"(.*?)"', ln)
            opts.extend(x.strip() for x in items if x.strip())
            break
        if re.match(r'^\s*reasoning\s*$', ln):
            continue
        items = re.findall(r'"(.*?)"', ln)
        opts.extend(x.strip() for x in items if x.strip())
    return opts

def extract_reasoning_as_rationale(s: str) -> Optional[str]:
    lines = s.splitlines()
    cap, buf = False, []
    for ln in lines:
        if re.match(r'^\s*reasoning\s*$', ln):
            cap = True
            continue
        if cap:
            if re.search(r'^\s*"[a-zA-Z_]+\s*":', ln):  # next key
                break
            if re.match(r'^\s*"\s*,?\s*$', ln):        # dangling quote line
                break
            buf.append(ln.rstrip())
    text = "\n".join(buf).strip()
    return text.strip('", ') if text else None

def normalize_schema(d: Dict[str, Any]) -> Dict[str, Any]:
    norm = {
        "question_type": d.get("question_type") or "Unknown",
        "question": d.get("question") or "",
        "options": d.get("options") if isinstance(d.get("options"), list) else [],
        "answer": d.get("answer") or "",
        "rationale": d.get("rationale") or "",
        "intended_skill": d.get("intended_skill") or "",
        "difficulty": d.get("difficulty") or "",
    }
    # pass-through (있으면 유지)
    for k in ("image_type", "image_file", "model"):
        if k in d: norm[k] = d[k]
    return norm

def build_human_value(question: Optional[str], options: Optional[List[str]]) -> str:
    q = (question or "").strip()
    base = "<image>\n" + q
    opts = [o.strip() for o in (options or []) if isinstance(o, str) and o.strip()]
    if opts:
        numbered = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(opts))
        base += "\nOptions are as follows:\n" + numbered
    else:
        base += "\nThere are no options."
    return base

def repair_inner_text_to_schema(txt: str) -> Dict[str, Any]:
    inner = strip_code_fence(txt)
    loaded = try_load(inner)
    if loaded is not None:
        return normalize_schema(loaded)

    qtype = extract_key_string("question_type", inner) or "Unknown"
    question = extract_key_string("question", inner) or ""
    answer = extract_key_string("answer", inner) or ""
    rationale = extract_key_string("rationale", inner)
    intended = extract_key_string("intended_skill", inner) or ""
    difficulty = extract_key_string("difficulty", inner) or ""
    options = extract_options_block(inner)
    if not rationale:
        rationale = extract_reasoning_as_rationale(inner) or ""

    return normalize_schema({
        "question_type": qtype,
        "question": question,
        "options": options,
        "answer": answer,
        "rationale": rationale,
        "intended_skill": intended,
        "difficulty": difficulty,
    })

def process_file(p: str) -> Optional[str]:
    """
    반환:
      - 수정 시: 설명 문자열
      - 미수정: None
    """
    raw = read_text(p)

    # 0) 에러-래퍼 감지: {"error": "...", "raw_response": "..."}
    wrapper = try_load(raw)
    if wrapper and isinstance(wrapper, dict) and "raw_response" in wrapper and "error" in wrapper:
        # raw_response 안쪽을 복구해서 같은 파일에 저장
        inner = str(wrapper.get("raw_response") or "")
        fixed = repair_inner_text_to_schema(inner)

        # image_file / model / image_type 같은 메타가 wrapper 쪽에 있으면 보존
        for k in ("image_type", "image_file", "model"):
            if k in wrapper and k not in fixed:
                fixed[k] = wrapper[k]

        backup_once(p)
        write_json(p, fixed)
        return "fixed: error-wrapper -> normalized schema"

    # 1) 일반 JSON: 파싱 성공이면 수정하지 않음 (구문 문제 아님)
    if wrapper is not None:
        return None

    # 2) 파싱 실패: 코드펜스/깨짐 복구
    fixed = repair_inner_text_to_schema(raw)
    backup_once(p)
    write_json(p, fixed)
    return "fixed: json-parse-failed -> repaired"

def main():
    root = os.path.abspath(JSON_ROOT)
    if not os.path.isdir(root):
        print(f"[error] JSON_ROOT not found: {JSON_ROOT}")
        return

    total, repaired, failed = 0, 0, 0
    logs: List[str] = []
    for r, _, files in os.walk(root):
        for fn in files:
            if not fn.lower().endswith(".json"):
                continue
            total += 1
            p = os.path.join(r, fn)
            try:
                msg = process_file(p)
                if msg:
                    repaired += 1
                    logs.append(f"{msg} -> {os.path.relpath(p)}")
            except Exception as e:
                failed += 1
                logs.append(f"[fail] {os.path.relpath(p)} :: {e}")

    print(f"[report] scanned: {total}, repaired: {repaired}, failed: {failed}")
    if logs:
        print("---- details ----")
        for m in logs:
            print(m)

if __name__ == "__main__":
    main()
