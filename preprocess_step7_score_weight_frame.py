import json
import numpy as np
from typing import List, Dict, Tuple

def read_jsonl(file_path: str) -> List[Dict]:
    """JSONL 파일을 읽어서 리스트로 반환"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def write_jsonl(data: List[Dict], file_path: str):
    """데이터를 JSONL 파일로 저장"""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def get_segment_ranges(segments: List[List[int]], not_segments: List[List[int]]) -> List[Tuple[int, int]]:
    """segments와 not_segments를 합쳐서 전체 구간 리스트 반환"""
    all_segments = []
    for seg in segments:
        all_segments.append((seg[0], seg[1]))
    for seg in not_segments:
        all_segments.append((seg[0], seg[1]))
    # 시작점 기준으로 정렬
    all_segments.sort(key=lambda x: x[0])
    return all_segments

def calculate_sample_distribution(scores: List[float], total_samples: int) -> List[int]:
    """
    점수를 정규화하여 각 구간에서 샘플링할 프레임 개수 계산
    """
    if not scores or total_samples == 0:
        return []
    
    # 점수가 1개인 경우
    if len(scores) == 1:
        return [total_samples]
    
    # 점수 정규화 (최소값을 0.1로 설정하여 모든 구간에서 최소한의 샘플링 보장)
    scores_array = np.array(scores)
    min_score = 0.1
    normalized_scores = np.maximum(scores_array, min_score)
    
    # 전체 합이 1이 되도록 정규화
    normalized_scores = normalized_scores / normalized_scores.sum()
    
    # 각 구간별 샘플 개수 계산
    sample_counts = (normalized_scores * total_samples).astype(int)
    
    # 반올림으로 인한 차이 보정
    diff = total_samples - sample_counts.sum()
    if diff > 0:
        # 부족한 샘플은 점수가 높은 구간에 추가
        top_indices = np.argsort(scores_array)[-diff:]
        for idx in top_indices:
            sample_counts[idx] += 1
    elif diff < 0:
        # 초과된 샘플은 점수가 낮은 구간에서 제거
        bottom_indices = np.argsort(scores_array)[:abs(diff)]
        for idx in bottom_indices:
            if sample_counts[idx] > 0:
                sample_counts[idx] -= 1
    
    # 최소 0 보장
    sample_counts = np.maximum(sample_counts, 0)
    
    return sample_counts.tolist()

def sample_frames_from_segment(start: int, end: int, n_samples: int) -> List[int]:
    """
    구간 내에서 균등하게 프레임 샘플링
    """
    # n_samples가 0이면 빈 리스트 반환
    if n_samples <= 0:
        return []
    
    segment_length = end - start + 1
    
    if segment_length <= n_samples:
        # 구간이 샘플 개수보다 작거나 같으면 모든 프레임 반환
        return list(range(start, end + 1))
    
    # 균등하게 샘플링
    step = segment_length / n_samples
    sampled = []
    for i in range(n_samples):
        # 각 구간의 중앙값을 선택
        pos = start + int(i * step + step / 2)
        sampled.append(min(pos, end))  # 범위를 벗어나지 않도록
    
    return sampled

def process_video_data(data: Dict, sample_sizes: List[int] = [8, 16, 32]) -> Dict:
    """
    비디오 데이터를 처리하여 각 샘플 크기별로 프레임 인덱스 계산
    """
    n_frames = data['n_frames']
    all_scores = data['all_scores']
    segments = data.get('segments', [])
    not_segments = data.get('not_segments', [])
    
    # 모든 구간 가져오기
    all_segments = get_segment_ranges(segments, not_segments)
    
    # 각 샘플 크기별로 처리
    for sample_size in sample_sizes:
        # 각 구간별 샘플 개수 계산
        sample_distribution = calculate_sample_distribution(all_scores, sample_size)
        
        # 키 이름 생성
        sample_number_key = f'all_scores_sample_frame_number_{sample_size}'
        sample_idx_key = f'all_scores_sample_frame_idx_{sample_size}'
        
        # 샘플 개수 저장
        data[sample_number_key] = sample_distribution
        
        # 각 구간별 샘플링된 프레임 인덱스 계산
        sampled_frames_dict = {}
        remaining_samples = 0
        segments_with_capacity = []
        
        for i, (seg_start, seg_end) in enumerate(all_segments):
            if i < len(sample_distribution):
                n_samples = sample_distribution[i]
                
                # n_samples가 0이면 스킵
                if n_samples == 0:
                    sampled_frames_dict[i] = []
                    continue
                    
                segment_length = seg_end - seg_start + 1
                
                if segment_length < n_samples:
                    # 구간이 샘플 개수보다 작은 경우
                    sampled = list(range(seg_start, seg_end + 1))
                    remaining_samples += n_samples - len(sampled)
                else:
                    sampled = sample_frames_from_segment(seg_start, seg_end, n_samples)
                    segments_with_capacity.append((i, seg_start, seg_end, all_scores[i]))
                
                sampled_frames_dict[i] = sampled
        
        # 남은 샘플이 있으면 점수가 높은 다른 구간에서 추가 샘플링
        if remaining_samples > 0 and segments_with_capacity:
            # 점수가 높은 순으로 정렬
            segments_with_capacity.sort(key=lambda x: x[3], reverse=True)
            
            for seg_idx, seg_start, seg_end, _ in segments_with_capacity:
                if remaining_samples <= 0:
                    break
                
                current_samples = sampled_frames_dict[seg_idx]
                segment_length = seg_end - seg_start + 1
                available_frames = set(range(seg_start, seg_end + 1)) - set(current_samples)
                
                if available_frames:
                    additional_samples = min(remaining_samples, len(available_frames))
                    # 균등하게 추가 샘플링
                    available_sorted = sorted(list(available_frames))
                    step = len(available_sorted) / additional_samples
                    for i in range(additional_samples):
                        idx = int(i * step + step / 2)
                        idx = min(idx, len(available_sorted) - 1)
                        current_samples.append(available_sorted[idx])
                    
                    current_samples.sort()
                    sampled_frames_dict[seg_idx] = current_samples
                    remaining_samples -= additional_samples
        
        # 결과 저장 (딕셔너리 형태로)
        data[sample_idx_key] = {str(k): v for k, v in sampled_frames_dict.items()}
    
    return data

def main(input_file: str = 'input.jsonl', output_file: str = 'result_score_idx.jsonl'):
    """
    메인 처리 함수
    """
    # JSONL 파일 읽기
    print(f"Reading {input_file}...")
    data_list = read_jsonl(input_file)
    
    # 각 비디오 데이터 처리
    print(f"Processing {len(data_list)} videos...")
    processed_data = []
    
    for i, video_data in enumerate(data_list):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(data_list)} videos...")
        
        processed = process_video_data(video_data.copy())
        processed_data.append(processed)
    
    # 결과 저장
    print(f"Writing results to {output_file}...")
    write_jsonl(processed_data, output_file)
    print("Done!")
    
    # 첫 번째 데이터 샘플 출력 (확인용)
    if processed_data:
        print("\nSample output for first video:")
        sample = processed_data[0]
        print(f"Video ID: {sample.get('video_id', 'N/A')}")
        print(f"Total frames: {sample.get('n_frames', 'N/A')}")
        print(f"All scores: {sample.get('all_scores', [])}")
        
        for sample_size in [8, 16, 32]:
            number_key = f'all_scores_sample_frame_number_{sample_size}'
            idx_key = f'all_scores_sample_frame_idx_{sample_size}'
            
            if number_key in sample and idx_key in sample:
                print(f"\nSample size {sample_size}:")
                print(f"  Distribution: {sample[number_key]}")
                print(f"  Sampled frames:")
                for seg_idx, frames in sample[idx_key].items():
                    print(f"    Segment {seg_idx}: {frames}")

if __name__ == "__main__":
    # 파일 경로를 필요에 따라 수정하세요
    main(input_file='result_score.jsonl', output_file='result_score_idx.jsonl')