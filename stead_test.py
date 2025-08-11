import torch
import numpy as np
import cv2
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
    
    def preprocess_video(self, video_path, n_frames=192, target_size=(16, 10, 10)):
        """
        비디오 전처리
        
        Args:
            video_path (str): 비디오 파일 경로
            n_frames (int): 샘플링할 프레임 수
            target_size (tuple): 목표 크기 (depth, height, width)
            
        Returns:
            torch.Tensor: 전처리된 비디오 텐서 [1, n_frames, depth, height, width]
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
        
        # 전체 프레임 수 확인
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < n_frames:
            print(f"경고: 비디오 프레임 수({total_frames})가 요구 프레임 수({n_frames})보다 적습니다.")
        
        # 프레임 인덱스 계산 (균등 샘플링)
        if total_frames > n_frames:
            frame_indices = np.linspace(0, total_frames-1, n_frames, dtype=int)
        else:
            # 프레임이 부족하면 반복해서 채움
            frame_indices = np.tile(np.arange(total_frames), 
                                  (n_frames // total_frames) + 1)[:n_frames]
        
        frames = []
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                # 프레임 읽기 실패시 이전 프레임 사용
                if frames:
                    frame = frames[-1]
                else:
                    continue
            
            # 프레임 전처리
            frame = cv2.resize(frame, (target_size[2], target_size[1]))  # (W, H)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0  # 정규화
            
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError("유효한 프레임을 읽을 수 없습니다.")
        
        # 부족한 프레임을 마지막 프레임으로 채움
        while len(frames) < n_frames:
            frames.append(frames[-1])
        
        # numpy array로 변환: [n_frames, H, W, C]
        video_array = np.stack(frames[:n_frames])
        
        # 차원 조정: [n_frames, H, W, C] -> [n_frames, C, H, W]
        video_array = video_array.transpose(0, 3, 1, 2)
        
        # depth 차원 추가 및 조정: [n_frames, C, H, W] -> [n_frames, depth, H, W]
        if target_size[0] != video_array.shape[1]:  # depth != channels
            # 간단한 방법: RGB를 depth 차원으로 확장
            video_array = np.repeat(video_array, 
                                  target_size[0] // video_array.shape[1] + 1, 
                                  axis=1)[:, :target_size[0], :, :]
        
        # 배치 차원 추가: [1, n_frames, depth, H, W]
        video_tensor = torch.from_numpy(video_array).unsqueeze(0).float()
        
        return video_tensor
    
    def predict(self, video_input):
        """
        이상 상황 예측
        
        Args:
            video_input: 비디오 파일 경로(str) 또는 전처리된 텐서(torch.Tensor)
            
        Returns:
            tuple: (anomaly_score, features)
                - anomaly_score (float): 이상 상황 확률 (0~1)
                - features (numpy.ndarray): 추출된 특징 벡터
        """
        # 입력이 문자열(파일 경로)인 경우 전처리
        if isinstance(video_input, str):
            video_tensor = self.preprocess_video(video_input)
        else:
            video_tensor = video_input
        
        # 모델 추론
        with torch.no_grad():
            video_tensor = video_tensor.to(self.device)
            scores, features = self.model(video_tensor)
            
            # 시그모이드 적용하여 확률로 변환
            anomaly_score = torch.sigmoid(scores).squeeze().cpu().item()
            features = features.squeeze().cpu().numpy()
        
        return anomaly_score, features
    
    def predict_batch(self, video_paths):
        """
        여러 비디오에 대한 배치 예측
        
        Args:
            video_paths (list): 비디오 파일 경로 리스트
            
        Returns:
            list: [(anomaly_score, features), ...] 형태의 결과 리스트
        """
        results = []
        for video_path in video_paths:
            try:
                result = self.predict(video_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                results.append((0.0, None))  # 에러시 기본값
        
        return results


# 사용 예시
def main():
    # 모델 초기화
    detector = VideoAnomalyDetector(
        model_path='assets/888tiny.pkl', #'ucf_crime/ucf-crime/videos/train/Abuse001_x264.mp4'
        model_arch='tiny',
        device='cuda'
    )
    
    # 단일 비디오 예측
    video_path = 'ucf_crime/ucf-crime/videos/train/Abuse001_x264.mp4'
    video_path = "ucf_crime/ucf-crime/videos/train/Normal_Videos211_x264.mp4"
    anomaly_score, features = detector.predict(video_path)
    
    print(f"이상 상황 확률: {anomaly_score:.4f}")
    print(f"특징 벡터 크기: {features.shape if features is not None else 'None'}")
    
    # 임계값 기반 판단
    threshold = 0.5
    if anomaly_score > threshold:
        print("⚠️  이상 상황 감지!")
    else:
        print("✅ 정상 상황")
    
    # # 배치 예측 예시
    # video_list = ["ucf_crime/ucf-crime/videos/train/Abuse001_x264.mp4", "ucf_crime/ucf-crime/videos/train/Abuse002_x264.mp4"]
    # batch_results = detector.predict_batch(video_list)
    
    # for i, (score, feat) in enumerate(batch_results):
    #     print(f"Video {i+1}: 이상 확률 {score:.4f}")


if __name__ == "__main__":
    main()
