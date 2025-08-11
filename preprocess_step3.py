import os
import json
from pathlib import Path
from decord import VideoReader, cpu
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np

def split_into_3_frames(start, end):
    if end <= start:
        return [start] * 3
    points = np.linspace(start, end, 4, dtype=int)[1:4]  # 1/4, 2/4, 3/4 지점
    return points.tolist()
def process_video(data, video_dir, save_dir):
    """단일 비디오 처리 함수 (멀티프로세싱용)"""
    video_id = data["video_id"]
    anomaly_ranges = data.get("anomaly_frame_ranges", [])
    segments = data.get("segments", [])

    # 비디오 파일 경로 확인
    video_path = os.path.join(video_dir, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        video_path = os.path.join(video_dir, f"{video_id}.avi")
    if not os.path.exists(video_path):
        return f"⚠ Video not found: {video_id}"

    # 저장 경로 생성
    video_save_dir = Path(save_dir) / video_id
    video_save_dir.mkdir(parents=True, exist_ok=True)

    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        n_frames = len(vr)

        # 1️⃣ anomaly 구간 처리
        for start, end in anomaly_ranges:
            if start >= n_frames:
                continue
            end = min(end, n_frames - 1)
            
            points = np.linspace(start, end, 4, dtype=int)[1:4]  # 3개만
            for frame_idx in points:
                frame = vr[frame_idx].asnumpy()
                save_name = f"anomaly_{frame_idx:06d}.jpg"
                save_path = video_save_dir / save_name
                cv2.imwrite(str(save_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        # 2️⃣ segment 구간 처리
        for start, end in segments:
            if start >= n_frames:
                continue
            end = min(end, n_frames - 1)
            center = (start + end) // 2
            frame = vr[center].asnumpy()
            save_name = f"segment_{center:06d}.jpg"
            save_path = video_save_dir / save_name
            cv2.imwrite(str(save_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        return f"✅ {video_id} processed"

    except Exception as e:
        return f"❌ Error {video_id}: {str(e)}"

def extract_frames_multiprocessing(jsonl_path: str, video_dir: str, save_dir: str, num_workers: int = None):
    os.makedirs(save_dir, exist_ok=True)

    # JSONL 로드
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data_list = [json.loads(line) for line in f if line.strip()]

    num_workers = num_workers or cpu_count()
    print(f"🔹 Using {num_workers} workers...")

    # process_video에 video_dir, save_dir 고정
    worker_fn = partial(process_video, video_dir=video_dir, save_dir=save_dir)
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(worker_fn, data_list),
            total=len(data_list),
            desc="Processing videos"
        ))

    # 처리 로그 요약
    print("\n".join(results))
    print("✅ All frames extracted.")


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--jsonl_path", required=True, help="result.jsonl 경로")
    # parser.add_argument("--video_dir", required=True, help="비디오 파일들이 있는 폴더 경로")
    # parser.add_argument("--save_dir", required=True, help="프레임 저장할 폴더 경로")
    # parser.add_argument("--num_workers", type=int, default=None, help="병렬 처리 프로세스 수 (기본: CPU 코어 수)")
    # args = parser.parse_args()

    # extract_frames_multiprocessing(
    #     args.jsonl_path,
    #     args.video_dir,
    #     args.save_dir,
    #     num_workers=args.num_workers
        
    # )
    extract_frames_multiprocessing(
        "/home/piawsa6000/nas192/datasets/projects/ICCE_2025/ICCE_2025/ucf_crime/result.jsonl",
        "/home/piawsa6000/nas192/datasets/projects/ICCE_2025/ICCE_2025/ucf_crime/ucf-crime/videos/train",
        "/home/piawsa6000/nas192/datasets/projects/ICCE_2025/ICCE_2025/result_bench_frame",
        8
    )
