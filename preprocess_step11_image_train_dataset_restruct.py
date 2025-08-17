#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
import glob
from typing import Any, Dict, List, Optional, Tuple

# =========================
# 하드코딩 설정 (원하는 값으로 수정하세요)
#   ※ 상대경로 권장 (현재 작업 디렉토리 기준)
# =========================
IMAGES_ROOT = "HAWK_bench"           # 이미지 루트 (예: HAWK_bench)
JSON_ROOT   = "HAWK_bench_json"      # JSON 루트 (폴더 순회 변환)
SINGLE_JSON_FILE = None              # 단일 파일 변환 시 JSON 경로, 사용 안하면 None
OUTPUT_FILE = "result_converted.jsonl"      # 출력 JSONL (상대경로 권장)
ID_START = 0                         # 시작 ID
# =========================


def to_cwd_rel(path: str) -> str:
    """절대/상대 어떤 입력이든 현재 작업 디렉토리 기준 상대경로로 반환."""
    try:
        return os.path.relpath(path, start=os.getcwd())
    except Exception:
        # 혹시 relpath 에러 시 원본 반환
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
    image_path: str,
    human_value: str,
    gpt_value: str
) -> Dict[str, Any]:
    # 반드시 현재 작업 디렉토리 기준 상대경로로 저장
    image_rel = to_cwd_rel(image_path)
    return {
        "id": _id,
        "type": "image",
        "task": task,
        "image": image_rel,
        "conversations": [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": gpt_value}
        ]
    }


def safe_join(*parts: str) -> str:
    return os.path.normpath(os.path.join(*parts))


def find_image_for_video(images_root: str, video_path: str) -> Tuple[str, List[str]]:
    """
    레거시 포맷에서 video 경로를 받아 가능한 이미지 경로를 추정.
    1) images_root 아래에서 비디오 basename과 같은 파일을 우선 검색.
    2) 없으면 확장자를 .jpg로 치환한 경로를 반환(없어도 경고만).
    반환: (image_path, warnings)
    """
    warns: List[str] = []
    video_base = os.path.splitext(os.path.basename(video_path))[0]
    exts = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]
    candidates = []
    for pattern in exts:
        candidates.extend(
            glob.glob(os.path.join(images_root, "**", video_base + pattern), recursive=True)
        )
    if candidates:
        return os.path.normpath(candidates[0]), warns

    guessed = os.path.join(images_root, video_base + ".jpg")
    if not os.path.exists(guessed):
        warns.append(f"[warn] Cannot find image for video '{video_path}'. Guessed path does not exist: {to_cwd_rel(guessed)}")
    return os.path.normpath(guessed), warns


def convert_hawk_record(
    data: Dict[str, Any],
    images_root: str,
    rel_dir: str,
    new_id: int
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    HAWK_bench_json 표준 레코드 변환
    """
    warns: List[str] = []
    question_type = data.get("question_type")
    question = data.get("question")
    options = data.get("options")  # None / [] / list[str]
    answer = data.get("answer")
    image_file = data.get("image_file")

    if not question_type:
        warns.append("[skip] hawk: missing 'question_type'")
        return None, warns
    if not image_file:
        warns.append("[skip] hawk: missing 'image_file'")
        return None, warns
    if answer is None:
        warns.append("[warn] Missing 'answer'; using empty string.")
        answer = ""

    image_path = safe_join(images_root, rel_dir, image_file)
    human_value = build_human_value(question or "", options)
    rec = to_jsonl_record(
        _id=new_id,
        task=str(question_type),
        image_path=image_path,
        human_value=human_value,
        gpt_value=str(answer)
    )
    return rec, warns


def convert_legacy_record(
    data: Dict[str, Any],
    images_root: str,
    new_id: int
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    레거시 포맷 변환:
    {"id":0,"type":"video","task":"judgement","video":"...","prompt":"...","anwser":"..."}
    규칙:
    - task: question_type 있으면 우선, 없으면 기존 task
    - options 없음 → "There are no options." 문구 포함
    - video → image 경로 추정
    """
    warns: List[str] = []
    task_value = data.get("question_type") or data.get("task") or "Unknown"
    prompt = data.get("prompt") or ""
    answer = data.get("answer", data.get("anwser", ""))

    video_path = data.get("video")
    if not video_path:
        warns.append("[skip] legacy: missing 'video'")
        return None, warns

    image_path, w = find_image_for_video(images_root, video_path)
    warns.extend(w)

    human_value = build_human_value(prompt, None)  # 레거시는 옵션 없음
    rec = to_jsonl_record(
        _id=new_id,
        task=str(task_value),
        image_path=image_path,
        human_value=human_value,
        gpt_value=str(answer)
    )
    return rec, warns


def walk_and_convert(
    images_root: str,
    json_root: str,
    out_path: str,
    id_start: int
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_f = open(out_path, "w", encoding="utf-8")
    count = 0
    warns_total: List[str] = []
    skip_details: List[str] = []

    for root, _, files in os.walk(json_root):
        rel_dir = os.path.relpath(root, json_root)
        if rel_dir == ".":
            rel_dir = ""
        for fn in files:
            if not fn.lower().endswith(".json"):
                continue
            src = os.path.join(root, fn)
            try:
                with open(src, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                msg = f"[skip] load-error: {e}"
                warns_total.append(f"[warn] Failed to load '{to_cwd_rel(src)}': {e}")
                skip_details.append(f"{msg} -> {to_cwd_rel(src)}")
                continue

            if "question_type" in data and "image_file" in data:
                rec, warns = convert_hawk_record(data, images_root, rel_dir, id_start + count)
            else:
                rec, warns = convert_legacy_record(data, images_root, id_start + count)

            # 누적 경고
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

    # 경고/스킵 요약
    if warns_total:
        sys.stderr.write("\n".join(warns_total) + "\n")
    if skip_details:
        sys.stderr.write("\n=== Skip Summary ===\n" + "\n".join(skip_details) + "\n")
    print(f"Done. Wrote {count} records to {to_cwd_rel(out_path)}")


def convert_single(
    images_root: str,
    json_file: str,
    out_path: str,
    id_start: int
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    skip_details: List[str] = []
    warns_total: List[str] = []

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        msg = f"[skip] load-error: {e}"
        skip_details.append(f"{msg} -> {to_cwd_rel(json_file)}")
        sys.stderr.write("\n".join(skip_details) + "\n")
        print("Done. Wrote 0 records.")
        return

    if "question_type" in data and ("image_file" in data or "image" in data):
        rel_dir = ""
        image_file = data.get("image_file")
        if image_file is None and data.get("image"):
            # 이미 변환된 포맷일 수도 있음
            image_path = data["image"]
            human_value = build_human_value(data.get("question", ""), data.get("options"))
            task_value = data.get("question_type", "Unknown")
            rec = to_jsonl_record(
                _id=id_start,
                task=str(task_value),
                image_path=image_path,
                human_value=human_value,
                gpt_value=str(data.get("answer", ""))
            )
        else:
            rec, warns = convert_hawk_record(data, images_root, rel_dir, id_start)
            for w in warns:
                if w.startswith("[skip]"):
                    skip_details.append(f"{w} -> {to_cwd_rel(json_file)}")
                else:
                    warns_total.append(f"{w} ({to_cwd_rel(json_file)})")
            if rec is None:
                if warns_total:
                    sys.stderr.write("\n".join(warns_total) + "\n")
                if skip_details:
                    sys.stderr.write("\n=== Skip Summary ===\n" + "\n".join(skip_details) + "\n")
                print("Done. Wrote 0 records.")
                return
    else:
        rec, warns = convert_legacy_record(data, images_root, id_start)
        for w in warns:
            if w.startswith("[skip]"):
                skip_details.append(f"{w} -> {to_cwd_rel(json_file)}")
            else:
                warns_total.append(f"{w} ({to_cwd_rel(json_file)})")
        if rec is None:
            if warns_total:
                sys.stderr.write("\n".join(warns_total) + "\n")
            if skip_details:
                sys.stderr.write("\n=== Skip Summary ===\n" + "\n".join(skip_details) + "\n")
            print("Done. Wrote 0 records.")
            return

    with open(out_path, "w", encoding="utf-8") as out_f:
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if warns_total:
        sys.stderr.write("\n".join(warns_total) + "\n")
    print(f"Done. Wrote 1 record to {to_cwd_rel(out_path)}")


def main():
    """
    argparse 없이 상단 하드코딩 설정으로만 동작.
    - SINGLE_JSON_FILE 가 설정되어 있으면 단일 파일 변환
    - 아니면 JSON_ROOT 를 순회하여 대량 변환
    - 출력/로그의 모든 경로는 현재 작업 디렉토리 기준 상대경로로 표기
    """
    if SINGLE_JSON_FILE:
        convert_single(IMAGES_ROOT, SINGLE_JSON_FILE, OUTPUT_FILE, ID_START)
    elif JSON_ROOT and os.path.isdir(JSON_ROOT):
        walk_and_convert(IMAGES_ROOT, JSON_ROOT, OUTPUT_FILE, ID_START)
    else:
        print("[error] No valid JSON input path. Set SINGLE_JSON_FILE or JSON_ROOT.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
