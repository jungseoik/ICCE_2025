import json
import pathlib

# ---------------------
# 0) 입·출력 경로 설정
# ---------------------
in_json  = pathlib.Path("ucf_database_train.json")   # 원본 JSON
out_jsonl = pathlib.Path("result.jsonl")             # 최종 jsonl

out_jsonl.parent.mkdir(parents=True, exist_ok=True)

# ---------------------
# 1) 원본 로드
# ---------------------
with in_json.open("r", encoding="utf-8") as f:
    db = json.load(f)

# ---------------------
# 2) 비디오별로 처리 & jsonl 저장
# ---------------------
with out_jsonl.open("w", encoding="utf-8") as fout:
    for vid, info in db.items():
        fps      = float(info["fps"])
        n_frames = int(info["n_frames"])
        events   = info.get("events", [])            # [[s,e], …]  (초 단위)

        # --- 구간별 판정 문장 확보 ---------------------------------
        if isinstance(info.get("events_summary_split"), list):
            # [{judgement: …}, …] 형태가 권장
            judgements = [d.get("judgement", "") for d in info["events_summary_split"]]
        else:
            # fallback: events_summary (문장 리스트)
            judgements = info.get("events_summary", [])

        # events ↔ judgements 길이가 다르면 안전하게 패딩
        if len(judgements) < len(events):
            judgements += [""] * (len(events) - len(judgements))
        
        # --- 이상 구간만 프레임 범위로 변환 --------------------------
        abnormal_ranges = []
        for (t_start, t_end), judge in zip(events, judgements):
            if "an anomaly exists" in judge.lower():   # 이상 판단
                f_start = max(0, int(round(t_start * fps)))
                f_end   = min(n_frames - 1, int(round(t_end   * fps)))
                abnormal_ranges.append([f_start, f_end])

        # --- jsonl 한 줄 작성 ---------------------------------------
        record = {
            "video_id": vid,
            "n_frames": n_frames,
            "anomaly_frame_ranges": abnormal_ranges   # 예: [[118,158],[942,1023]]
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Saved {out_jsonl.resolve()}  ({len(db)} videos)")
