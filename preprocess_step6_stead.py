import torch
import numpy as np
import cv2
import decord
from decord import VideoReader, cpu
import json
import os
from tqdm import tqdm
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

# 기존 VideoAnomalyDetector 클래스는 그대로 사용
from models.stead_model import Model
import torch.nn.functional as F

class VideoAnomalyDetector:
    def __init__(self, model_path, model_arch='tiny', device='cuda'):
        """
        비디오 이상 상황 탐지 모델 래퍼
        
        Args:
            model_path (str): 모델 가중치 파일 경로 (.pkl)
            model_arch (str): 모델 아키텍처 ('base', 'fast', 'tiny')
            device (str): 사용할 디바이스 ('cuda', 'cpu')
        """
        self.device = torch.device(device)
        self.model = self._load_model(model_path, model_arch)
        
    def _load_model(self, model_path, model_arch):
        """모델 로드"""
        if model_arch == 'base':
            model = Model()
        elif model_arch in ['fast', 'tiny']:
            model = Model(ff_mult=1, dims=(32,32), depths=(1,1))
        else:
            raise ValueError(f"Unknown model architecture: {model_arch}")
            
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def preprocess_video_tensor(self, video_frames, n_frames=192, target_size=(16, 10, 10)):
        """
        비디오 프레임 텐서 전처리
        
        Args:
            video_frames (np.ndarray): 비디오 프레임 배열 [T, H, W, C]
            n_frames (int): 샘플링할 프레임 수
            target_size (tuple): 목표 크기 (depth, height, width)
            
        Returns:
            torch.Tensor: 전처리된 비디오 텐서 [1, n_frames, depth, height, width]
        """
        total_frames = len(video_frames)
        
        # 프레임 인덱스 계산 (균등 샘플링)
        if total_frames > n_frames:
            frame_indices = np.linspace(0, total_frames-1, n_frames, dtype=int)
        else:
            # 프레임이 부족하면 반복해서 채움
            frame_indices = np.tile(np.arange(total_frames), 
                                  (n_frames // total_frames) + 1)[:n_frames]
        
        frames = []
        for frame_idx in frame_indices:
            frame = video_frames[frame_idx]
            
            # 프레임 전처리
            if frame.shape[:2] != (target_size[1], target_size[2]):
                frame = cv2.resize(frame, (target_size[2], target_size[1]))  # (W, H)
            
            # BGR to RGB 변환 (OpenCV는 BGR, decord는 RGB)
            if frame.shape[-1] == 3:  # RGB 채널이 있는 경우
                frame = frame.astype(np.float32) / 255.0  # 정규화
            
            frames.append(frame)
        
        if len(frames) == 0:
            raise ValueError("유효한 프레임을 읽을 수 없습니다.")
        
        # 부족한 프레임을 마지막 프레임으로 채움
        while len(frames) < n_frames:
            frames.append(frames[-1])
        
        # numpy array로 변환: [n_frames, H, W, C]
        video_array = np.stack(frames[:n_frames])
        
        # 차원 조정: [n_frames, H, W, C] -> [n_frames, C, H, W]
        video_array = video_array.transpose(0, 3, 1, 2)
        
        # depth 차원 추가 및 조정
        if target_size[0] != video_array.shape[1]:
            video_array = np.repeat(video_array, 
                                  target_size[0] // video_array.shape[1] + 1, 
                                  axis=1)[:, :target_size[0], :, :]
        
        # 배치 차원 추가: [1, n_frames, depth, H, W]
        video_tensor = torch.from_numpy(video_array).unsqueeze(0).float()
        
        return video_tensor
    
    def predict_tensor(self, video_tensor):
        """
        이상 상황 예측 (텐서 입력)
        
        Args:
            video_tensor (torch.Tensor): 전처리된 비디오 텐서
            
        Returns:
            tuple: (anomaly_score, features)
        """
        with torch.no_grad():
            video_tensor = video_tensor.to(self.device)
            scores, features = self.model(video_tensor)
            
            # 시그모이드 적용하여 확률로 변환
            anomaly_score = torch.sigmoid(scores).squeeze().cpu().item()
            features = features.squeeze().cpu().numpy()
        
        return anomaly_score, features
    
    def predict_batch_tensors(self, video_tensors):
        """
        배치로 여러 텐서 예측 (GPU 최적화)
        
        Args:
            video_tensors (list): 전처리된 비디오 텐서 리스트
            
        Returns:
            list: 이상상황 스코어 리스트
        """
        if not video_tensors:
            return []
        
        # 텐서들을 배치로 결합
        batch_tensor = torch.cat(video_tensors, dim=0).to(self.device)
        
        with torch.no_grad():
            scores, _ = self.model(batch_tensor)
            # 시그모이드 적용하여 확률로 변환
            anomaly_scores = torch.sigmoid(scores).squeeze().cpu().tolist()
            
            # 단일 스코어인 경우 리스트로 변환
            if not isinstance(anomaly_scores, list):
                anomaly_scores = [anomaly_scores]
        
        return anomaly_scores


class SegmentAnomalyProcessor:
    def __init__(self, model_path, model_arch='tiny', device='cuda', video_base_path='', batch_size=8):
        """
        세그먼트별 이상상황 스코어 계산기
        
        Args:
            model_path (str): 모델 가중치 파일 경로
            model_arch (str): 모델 아키텍처
            device (str): 사용할 디바이스
            video_base_path (str): 비디오 파일들의 기본 경로
            batch_size (int): GPU 배치 크기
        """
        self.detector = VideoAnomalyDetector(model_path, model_arch, device)
        self.video_base_path = video_base_path
        self.batch_size = batch_size
        
    def calculate_not_segments(self, total_frames, segments):
        """
        전체 프레임에서 segments를 제외한 나머지 구간들 계산
        
        Args:
            total_frames (int): 전체 프레임 수
            segments (list): 기존 세그먼트 리스트 [[start, end], ...]
            
        Returns:
            list: not_segments 리스트 [[start, end], ...]
        """
        if not segments:
            return [[0, total_frames - 1]]
        
        # 세그먼트들을 시작점 기준으로 정렬
        segments = sorted(segments, key=lambda x: x[0])
        
        not_segments = []
        current_pos = 0
        
        for start, end in segments:
            # 현재 위치와 세그먼트 시작점 사이에 gap이 있으면 not_segment 추가
            if current_pos < start:
                not_segments.append([current_pos, start - 1])
            
            # 다음 위치 업데이트
            current_pos = max(current_pos, end + 1)
        
        # 마지막 세그먼트 이후에 프레임이 남아있으면 not_segment 추가
        if current_pos < total_frames:
            not_segments.append([current_pos, total_frames - 1])
        
        return not_segments
    
    def extract_segment_frames_decord(self, video_path, start_frame, end_frame):
        """
        Decord를 사용하여 특정 세그먼트의 프레임 추출
        
        Args:
            video_path (str): 비디오 파일 경로
            start_frame (int): 시작 프레임 인덱스
            end_frame (int): 종료 프레임 인덱스
            
        Returns:
            np.ndarray: 프레임 배열 [T, H, W, C]
        """
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            
            # 프레임 인덱스 조정
            start_frame = max(0, start_frame)
            end_frame = min(total_frames - 1, end_frame)
            
            if start_frame >= end_frame:
                raise ValueError(f"Invalid frame range: {start_frame}-{end_frame}")
            
            # 프레임 추출
            frame_indices = list(range(start_frame, end_frame + 1))
            frames = vr.get_batch(frame_indices).asnumpy()  # [T, H, W, C]
            
            return frames
            
        except Exception as e:
            print(f"Decord 추출 실패: {e}")
            return self.extract_segment_frames_opencv(video_path, start_frame, end_frame)
    
    def extract_segment_frames_opencv(self, video_path, start_frame, end_frame):
        """
        OpenCV를 사용하여 특정 세그먼트의 프레임 추출 (fallback)
        
        Args:
            video_path (str): 비디오 파일 경로
            start_frame (int): 시작 프레임 인덱스
            end_frame (int): 종료 프레임 인덱스
            
        Returns:
            np.ndarray: 프레임 배열 [T, H, W, C]
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = max(0, start_frame)
        end_frame = min(total_frames - 1, end_frame)
        
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            
            # BGR to RGB 변환
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError("유효한 프레임을 추출할 수 없습니다.")
        
        return np.array(frames)  # [T, H, W, C]
    
    def extract_all_segments_parallel(self, video_path, all_segments):
        """
        모든 세그먼트의 프레임을 병렬로 추출
        
        Args:
            video_path (str): 비디오 파일 경로
            all_segments (list): 모든 세그먼트 리스트
            
        Returns:
            list: 추출된 프레임 배열들의 리스트
        """
        def extract_single_segment(segment):
            start_frame, end_frame = segment
            return self.extract_segment_frames_decord(video_path, start_frame, end_frame)
        
        # 병렬 처리로 모든 세그먼트 추출
        with ThreadPoolExecutor(max_workers=4) as executor:
            frames_list = list(executor.map(extract_single_segment, all_segments))
        
        return frames_list
    
    def calculate_all_scores(self, video_data):
        """
        특정 비디오의 모든 구간(segments + not_segments)에 대한 이상상황 스코어 계산
        
        Args:
            video_data (dict): 비디오 정보 (video_id, n_frames, segments 등)
            
        Returns:
            dict: 계산된 스코어들과 구간 정보
        """
        video_id = video_data['video_id']
        n_frames = video_data['n_frames']
        segments = video_data['segments']
        
        # 비디오 파일 경로 구성
        video_path = os.path.join(self.video_base_path, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            print(f"비디오 파일을 찾을 수 없습니다: {video_path}")
            return {
                'anomaly_segment_score': [0.0] * len(segments),
                'not_segments': [],
                'not_segment_score': [],
                'all_scores': [0.0] * len(segments)
            }
        
        # not_segments 계산
        not_segments = self.calculate_not_segments(n_frames, segments)
        
        # 모든 구간을 하나의 리스트로 결합 (순서와 타입 정보 유지)
        all_segments_with_type = []
        for i, segment in enumerate(segments):
            all_segments_with_type.append((segment, 'segment', i))
        for i, segment in enumerate(not_segments):
            all_segments_with_type.append((segment, 'not_segment', i))
        
        # 시간 순서대로 정렬
        all_segments_with_type.sort(key=lambda x: x[0][0])
        
        try:
            # 모든 세그먼트의 프레임 추출
            all_segments = [item[0] for item in all_segments_with_type]
            frames_list = self.extract_all_segments_parallel(video_path, all_segments)
            
            # 텐서로 변환 (배치 처리용)
            video_tensors = []
            valid_indices = []
            
            for i, frames in enumerate(frames_list):
                if frames is not None and len(frames) > 0:
                    try:
                        tensor = self.detector.preprocess_video_tensor(frames)
                        video_tensors.append(tensor)
                        valid_indices.append(i)
                    except Exception as e:
                        print(f"텐서 변환 오류: {e}")
            
            # 배치로 예측 (GPU 최적화)
            if video_tensors:
                # 배치 크기에 따라 나누어 처리
                all_scores_raw = []
                for i in range(0, len(video_tensors), self.batch_size):
                    batch = video_tensors[i:i+self.batch_size]
                    batch_scores = self.detector.predict_batch_tensors(batch)
                    all_scores_raw.extend(batch_scores)
            else:
                all_scores_raw = []
            
            # 결과 정리
            segment_scores = [0.0] * len(segments)
            not_segment_scores = [0.0] * len(not_segments)
            all_scores = [0.0] * len(all_segments_with_type)
            
            # 유효한 인덱스에 대해서만 스코어 할당
            score_idx = 0
            for i, (segment, seg_type, original_idx) in enumerate(all_segments_with_type):
                if i in valid_indices and score_idx < len(all_scores_raw):
                    score = all_scores_raw[score_idx]
                    score_idx += 1
                    
                    all_scores[i] = score
                    
                    if seg_type == 'segment':
                        segment_scores[original_idx] = score
                    else:  # not_segment
                        not_segment_scores[original_idx] = score
                    
                    print(f"{video_id} - {seg_type.upper()} [{segment[0]}, {segment[1]}]: Score = {score:.4f}")
            
            # 상세 결과 출력
            print(f"\n=== {video_id} 결과 요약 ===")
            print(f"Segments: {segments}")
            print(f"Segment Scores: {[f'{s:.4f}' for s in segment_scores]}")
            print(f"Not-Segments: {not_segments}")
            print(f"Not-Segment Scores: {[f'{s:.4f}' for s in not_segment_scores]}")
            print(f"All Scores (시간순): {[f'{s:.4f}' for s in all_scores]}")
            print("=" * 50)
            
            return {
                'anomaly_segment_score': segment_scores,
                'not_segments': not_segments,
                'not_segment_score': not_segment_scores,
                'all_scores': all_scores
            }
            
        except Exception as e:
            print(f"비디오 처리 오류 ({video_id}): {e}")
            return {
                'anomaly_segment_score': [0.0] * len(segments),
                'not_segments': not_segments,
                'not_segment_score': [0.0] * len(not_segments),
                'all_scores': [0.0] * (len(segments) + len(not_segments))
            }
    
    def process_jsonl(self, input_jsonl_path, output_jsonl_path):
        """
        JSONL 파일을 읽어서 모든 구간의 스코어를 계산하고 새 파일로 저장
        
        Args:
            input_jsonl_path (str): 입력 JSONL 파일 경로
            output_jsonl_path (str): 출력 JSONL 파일 경로
        """
        results = []
        
        # JSONL 파일 읽기
        with open(input_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                results.append(data)
        
        print(f"총 {len(results)}개의 비디오 처리 시작...")
        print(f"GPU 배치 크기: {self.batch_size}")
        
        # 각 비디오에 대해 모든 구간의 스코어 계산
        for i, video_data in enumerate(tqdm(results, desc="Processing videos")):
            try:
                print(f"\n[{i+1}/{len(results)}] 처리 중: {video_data['video_id']}")
                scores_data = self.calculate_all_scores(video_data)
                
                # 결과를 video_data에 추가
                video_data.update(scores_data)
                
                # JSONL에 저장될 최종 데이터 출력
                print(f"\n📝 JSONL 저장 데이터:")
                print(f"  - anomaly_segment_score: {scores_data['anomaly_segment_score']}")
                print(f"  - not_segments: {scores_data['not_segments']}")
                print(f"  - not_segment_score: {scores_data['not_segment_score']}")
                print(f"  - all_scores: {scores_data['all_scores']}")
                print("-" * 70)
                
            except Exception as e:
                print(f"비디오 {video_data['video_id']} 처리 중 오류: {e}")
                # 오류시 기본값 설정
                segments = video_data.get('segments', [])
                default_data = {
                    'anomaly_segment_score': [0.0] * len(segments),
                    'not_segments': [],
                    'not_segment_score': [],
                    'all_scores': [0.0] * len(segments)
                }
                video_data.update(default_data)
                
                print(f"📝 JSONL 저장 데이터 (오류시 기본값):")
                print(f"  - anomaly_segment_score: {default_data['anomaly_segment_score']}")
                print(f"  - not_segments: {default_data['not_segments']}")
                print(f"  - not_segment_score: {default_data['not_segment_score']}")
                print(f"  - all_scores: {default_data['all_scores']}")
                print("-" * 70)
        
        # 결과를 새 JSONL 파일로 저장
        with open(output_jsonl_path, 'w', encoding='utf-8') as f:
            for data in results:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        print(f"결과가 {output_jsonl_path}에 저장되었습니다.")


def main():
    # 설정
    model_path = 'assets/888tiny.pkl'
    model_arch = 'tiny'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    video_base_path = 'ucf-crime/videos/train'  # 비디오 파일들이 있는 기본 경로
    batch_size = 16  # GPU 메모리에 따라 조정
    
    input_jsonl = 'result.jsonl'
    output_jsonl = 'result_score.jsonl'
    
    print(f"사용 디바이스: {device}")
    if device == 'cuda':
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # 처리기 초기화
    processor = SegmentAnomalyProcessor(
        model_path=model_path,
        model_arch=model_arch,
        device=device,
        video_base_path=video_base_path,
        batch_size=batch_size
    )
    
    # JSONL 처리
    processor.process_jsonl(input_jsonl, output_jsonl)
    
    print("모든 구간의 이상상황 스코어 계산 완료!")


if __name__ == "__main__":
    main()