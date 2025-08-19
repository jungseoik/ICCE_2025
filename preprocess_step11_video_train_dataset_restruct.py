#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# =========================
# 하드코딩 설정 (원하는 값으로 수정하세요)
#   ※ 상대경로 권장 (현재 작업 디렉토리 기준)
# =========================
VIDEO_ROOT  = "ucf-crime/events/train"      # 비디오 파일들이 있는 루트 폴더
JSON_ROOT   = "IntenVL3_Violence_SFT/HIVAU-70k/train_total_video_image/HAWK_bench_video_json"       # 비디오 JSON 파일들이 있는 루트 폴더
SINGLE_JSON_FILE = None                     # 단일 파일 변환 시 JSON 경로, 사용 안하면 None
OUTPUT_FILE = "result_video_converted.jsonl"      # 출력 JSONL (상대경로 권장)
ID_START = 0                                # 시작 ID
# =========================


def to_cwd_rel(path: str) -> str:
    """절대/상대 어떤 입력이든 현재 작업 디렉토리 기준 상대경로로 반환."""
    try:
        return os.path.relpath(path, start=os.getcwd())
    except Exception:
        return path


def build_human_value(question: Optional[str], options: Optional[List[str]]) -> str:
    """
    <image>\n + question + (option text)
    - If options exist: "Options are as follows:\n1. ...\n2. ..."
    - If no options: "There are no options."
    """
    q = (question or "").rstrip()
    base = "<image>\n" + q

    has_options = bool(options) and any(isinstance(o, str) and o.strip() for o in options or [])
    if has_options:
        filtered = [o.strip() for o in options if isinstance(o, str) and o.strip()]
        numbered = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(filtered))
        base += "\nOptions are as follows:\n" + numbered
    else:
        base += "\nThere are no options."

    return base


def to_jsonl_record(
    _id: int,
    task: str,
    video_path: str,
    human_value: str,
    gpt_value: str
) -> Dict[str, Any]:
    """최종 JSONL 레코드를 생성."""
    video_rel = to_cwd_rel(video_path)
    return {
        "id": _id,
        "type": "event",  # 최종 포맷의 타입은 'image'로 유지
        "task": task,
        "video": video_rel, # key 이름은 'image'로 유지
        "conversations": [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": gpt_value}
        ]
    }


def safe_join(*parts: str) -> str:
    return os.path.normpath(os.path.join(*parts))


def convert_video_record(
    data: Dict[str, Any],
    video_root: str,
    new_id: int
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    비디오 JSON 레코드를 변환.
    """
    warns: List[str] = []
    question_type = data.get("question_type")
    question = data.get("question")
    options = data.get("options")
    answer = data.get("answer")
    video_file = data.get("video_file")

    if not question_type:
        warns.append("[skip] missing 'question_type'")
        return None, warns
    if not video_file:
        warns.append("[skip] missing 'video_file'")
        return None, warns
    if answer is None:
        warns.append("[warn] Missing 'answer'; using empty string.")
        answer = ""

    # video_root와 video_file을 조합하여 최종 경로 생성
    final_video_path = safe_join(video_root, video_file)
    
    human_value = build_human_value(question or "", options)
    rec = to_jsonl_record(
        _id=new_id,
        task=str(question_type),
        video_path=final_video_path,
        human_value=human_value,
        gpt_value=str(answer)
    )
    return rec, warns


def walk_and_convert(
    video_root: str,
    json_root: str,
    out_path: str,
    id_start: int
) -> None:
    """폴더를 순회하며 모든 JSON 파일을 변환."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_f = open(out_path, "w", encoding="utf-8")
    count = 0
    warns_total: List[str] = []
    skip_details: List[str] = []

    # JSON_ROOT 내의 모든 .json 파일을 찾음 (하위 폴더는 고려하지 않음)
    json_files = [f for f in os.listdir(json_root) if f.lower().endswith(".json")]

    for fn in json_files:
        src = os.path.join(json_root, fn)
        try:
            with open(src, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            msg = f"[skip] load-error: {e}"
            warns_total.append(f"[warn] Failed to load '{to_cwd_rel(src)}': {e}")
            skip_details.append(f"{msg} -> {to_cwd_rel(src)}")
            continue

        # 비디오 레코드 변환 함수만 직접 호출
        rec, warns = convert_video_record(data, video_root, id_start + count)

        for w in warns:
            if w.startswith("[skip]"):
                skip_details.append(f"{w} -> {to_cwd_rel(src)}")
            else:
                warns_total.append(f"{w} ({to_cwd_rel(src)})")

        if rec is None:
            continue

        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        count += 1

    out_f.close()

    if warns_total:
        sys.stderr.write("\n".join(warns_total) + "\n")
    if skip_details:
        sys.stderr.write("\n=== Skip Summary ===\n" + "\n".join(skip_details) + "\n")
    print(f"Done. Wrote {count} records to {to_cwd_rel(out_path)}")


def convert_single(
    video_root: str,
    json_file: str,
    out_path: str,
    id_start: int
) -> None:
    """단일 JSON 파일을 변환."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        sys.stderr.write(f"[error] Failed to load '{to_cwd_rel(json_file)}': {e}\n")
        print("Done. Wrote 0 records.")
        return

    rec, warns = convert_video_record(data, video_root, id_start)

    warns_total = [f"{w} ({to_cwd_rel(json_file)})" for w in warns]

    if rec is None:
        if warns_total:
            sys.stderr.write("\n".join(warns_total) + "\n")
        print("Done. Wrote 0 records.")
        return

    with open(out_path, "w", encoding="utf-8") as out_f:
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if warns_total:
        sys.stderr.write("\n".join(warns_total) + "\n")
    print(f"Done. Wrote 1 record to {to_cwd_rel(out_path)}")


def main():
    if SINGLE_JSON_FILE:
        convert_single(VIDEO_ROOT, SINGLE_JSON_FILE, OUTPUT_FILE, ID_START)
    elif JSON_ROOT and os.path.isdir(JSON_ROOT):
        walk_and_convert(VIDEO_ROOT, JSON_ROOT, OUTPUT_FILE, ID_START)
    else:
        print(f"[error] No valid JSON input path. Set SINGLE_JSON_FILE or check if '{JSON_ROOT}' directory exists.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()