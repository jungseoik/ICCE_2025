import os
import json
import pickle
import numpy as np
import torch
import ffmpeg
from pathlib import Path
from tqdm import tqdm
from utils.utils import get_batches, get_frames, predictions_to_scenes
from models.autoshot import TransNetV2Supernet
from assets.config import TRAIN_FOLDER, AUTOSHOT_MODEL_PATH, AUTOSHOT_THRESHOLD, RESULT_JSONL
class VideoSegmentationProcessor:
    def __init__(self, model_path="./ckpt_0_200_0.pth", threshold=0.296):
        """
        비디오 세그멘테이션 프로세서 초기화
        
        Args:
            model_path: 학습된 모델 경로
            threshold: 장면 경계 감지 임계값
        """
        self.threshold = threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 모델 로드 (주석 처리된 부분을 참고하여 실제 모델 로드)
        self.model = self._load_model(model_path)
        print(f"Using device: {self.device}")
        
    def _load_model(self, model_path):
        """모델 로드 (실제 모델 클래스가 있을 때 사용)"""
        model = TransNetV2Supernet().eval()
        
        if os.path.exists(model_path):
            print(f'Loading model from {model_path}')
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_path, map_location=self.device)
            pretrained_dict = {k: v for k, v in pretrained_dict['net'].items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            raise Exception(f"Model file not found: {model_path}")
        
        if self.device == "cuda":
            model = model.cuda(0)
        model.eval()
        return model
    
    def predict_video_segments(self, video_path):
        """
        비디오에서 장면 세그먼트 예측
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            list: 장면 구간 리스트 [[start1, end1], [start2, end2], ...]
        """
        # 프레임 추출
        frames = get_frames(video_path)
        if frames is None:
            return []
        
        # 모델이 없는 경우 더미 데이터 반환 (테스트용)
        if self.model is None:
            print(f"Warning: No model loaded. Returning dummy segments for {video_path}")
            # 더미 세그먼트: 비디오를 3등분
            n_frames = len(frames)
            segment_size = n_frames // 3
            return [
                [0, segment_size],
                [segment_size + 1, segment_size * 2],
                [segment_size * 2 + 1, n_frames - 1]
            ]
        
        # 실제 모델 예측
        predictions = []
        for batch in get_batches(frames):
            batch_pred = self._predict_batch(batch)
            predictions.append(batch_pred[25:75])
        
        predictions = np.concatenate(predictions, 0)[:len(frames)]
        
        # 임계값 적용하여 이진 예측으로 변환
        binary_predictions = (predictions > self.threshold).astype(np.uint8)
        
        # 장면 구간으로 변환
        scenes = predictions_to_scenes(binary_predictions)
        
        return scenes.tolist()
    
    def _predict_batch(self, batch):
        """배치 예측 (실제 모델이 있을 때 사용)"""
        batch = torch.from_numpy(batch.transpose((3, 0, 1, 2))[np.newaxis, ...]) * 1.0
        batch = batch.to(self.device)
        with torch.no_grad():
            one_hot = self.model(batch)
            if isinstance(one_hot, tuple):
                one_hot = one_hot[0]
            return torch.sigmoid(one_hot[0]).detach().cpu().numpy()
        
    def get_video_info(self, video_path):
        """비디오의 기본 정보 추출"""
        try:
            frames = get_frames(video_path)
            if frames is None:
                return None, 0
                
            video_id = Path(video_path).stem
            n_frames = len(frames)
            return video_id, n_frames
        except Exception as e:
            print(f"Error getting video info for {video_path}: {e}")
            return None, 0

def process_videos_and_update_jsonl(
    train_folder_path, 
    result_jsonl_path, 
    model_path="./ckpt_0_200_0.pth",
    threshold=0.296
):
    """
    train 폴더의 비디오들을 처리하고 result.jsonl 파일을 업데이트
    
    Args:
        train_folder_path: 비디오들이 있는 폴더 경로
        result_jsonl_path: 기존 result.jsonl 파일 경로
        model_path: 모델 파일 경로
        threshold: 장면 경계 감지 임계값
    """
    
    # 비디오 세그멘테이션 프로세서 초기화
    processor = VideoSegmentationProcessor(model_path, threshold)
    
    # 기존 result.jsonl 파일 읽기
    existing_data = {}
    if os.path.exists(result_jsonl_path):
        print(f"Loading existing data from {result_jsonl_path}")
        with open(result_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    existing_data[data['video_id']] = data
        print(f"Loaded {len(existing_data)} existing entries")
    
    # train 폴더에서 비디오 파일들 찾기
    train_folder = Path(train_folder_path)
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(train_folder.glob(f'*{ext}'))
    
    print(f"Found {len(video_files)} video files in {train_folder_path}")
    
    # 각 비디오 처리
    updated_count = 0
    error_count = 0
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        try:
            video_id, n_frames = processor.get_video_info(str(video_path))
            
            if video_id is None:
                print(f"Skipping {video_path} - could not extract video info")
                error_count += 1
                continue
            
            # 장면 세그먼트 예측
            segments = processor.predict_video_segments(str(video_path))
            
            # 기존 데이터에 세그먼트 정보 추가
            if video_id in existing_data:
                existing_data[video_id]['segments'] = segments
                print(f"Updated {video_id} with {len(segments)} segments")
            else:
                # 새로운 엔트리 생성
                existing_data[video_id] = {
                    'video_id': video_id,
                    'n_frames': n_frames,
                    'segments': segments
                }
                print(f"Created new entry for {video_id} with {len(segments)} segments")
            
            updated_count += 1
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            error_count += 1
            continue
    
    # 업데이트된 데이터를 result.jsonl에 저장
    backup_path = result_jsonl_path + '.backup'
    if os.path.exists(result_jsonl_path):
        os.rename(result_jsonl_path, backup_path)
        print(f"Backup created: {backup_path}")
    
    with open(result_jsonl_path, 'w', encoding='utf-8') as f:
        for video_id, data in existing_data.items():
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {updated_count} videos")
    print(f"Errors: {error_count} videos")
    print(f"Updated result.jsonl saved to: {result_jsonl_path}")
    print(f"Total entries in result.jsonl: {len(existing_data)}")

# 사용 예시
if __name__ == "__main__":

    process_videos_and_update_jsonl(
        train_folder_path=TRAIN_FOLDER,
        result_jsonl_path=RESULT_JSONL,
        model_path=AUTOSHOT_MODEL_PATH,
        threshold=AUTOSHOT_THRESHOLD
    )