#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

# ===== 설정 (원하면 수정) =====
FILE_A = "merge_instruction_train_final_only_ucf.jsonl"  # video 쪽
FILE_B = "result_converted.jsonl"                        # image 쪽
OUTPUT = "result_total_image_video.jsonl"                # 합친 결과
# ============================

def read_jsonl(path):
    items, errors = [], []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except Exception as e:
                errors.append((i, str(e)))
    return items, errors

def to_int_id(x):
    # id가 정수 아닌 경우 최대한 정수로 변환, 실패하면 None
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        try:
            return int(x)
        except Exception:
            return None
    return None

def main():
    # 1) 파일 A 읽기
    a_items, a_errs = read_jsonl(FILE_A)
    if a_errs:
        print(f"[warn] {FILE_A} parse errors: {len(a_errs)}")
        for ln, msg in a_errs[:10]:
            print(f"  line {ln}: {msg}")

    # 2) A의 max id 구하기 (정수인 것만)
    max_id = -1
    for obj in a_items:
        iid = to_int_id(obj.get("id"))
        if iid is not None and iid > max_id:
            max_id = iid

    # 3) 파일 B 읽기
    b_items, b_errs = read_jsonl(FILE_B)
    if b_errs:
        print(f"[warn] {FILE_B} parse errors: {len(b_errs)}")
        for ln, msg in b_errs[:10]:
            print(f"  line {ln}: {msg}")

    # 4) B의 id 재부여: A의 max_id 다음부터
    next_id = max_id + 1
    reassigned = 0
    for obj in b_items:
        obj["id"] = next_id
        next_id += 1
        reassigned += 1

    # 5) 출력 저장 (A 원본 그대로 + B 재할당)
    os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as out:
        for obj in a_items:
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        for obj in b_items:
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[done] wrote: {OUTPUT}")
    print(f"  A records: {len(a_items)} (max id in A = {max_id})")
    print(f"  B records: {len(b_items)} → reassigned: {reassigned}")
    print(f"  total: {len(a_items) + len(b_items)}")

if __name__ == "__main__":
    main()
