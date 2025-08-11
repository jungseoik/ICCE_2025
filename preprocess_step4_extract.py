import os
import shutil
import re
from pathlib import Path

def extract_number_from_filename(filename):
    """파일명에서 숫자를 추출합니다."""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

def copy_selected_images(source_folder, destination_folder):
    """
    소스 폴더에서 이미지를 선택적으로 복사합니다.
    
    Args:
        source_folder (str): 소스 폴더 경로 (비디오 폴더들이 있는 상위 폴더)
        destination_folder (str): 목적지 폴더 경로
    """
    source_path = Path(source_folder)
    dest_path = Path(destination_folder)
    
    if not source_path.exists():
        print(f"❌ 소스 폴더가 존재하지 않습니다: {source_folder}")
        return
    
    # 목적지 폴더 생성
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # 소스 폴더의 모든 하위 폴더 탐색
    video_folders = [folder for folder in source_path.iterdir() if folder.is_dir()]
    
    if not video_folders:
        print("❌ 비디오 폴더를 찾을 수 없습니다.")
        return
    
    print(f"📁 총 {len(video_folders)}개의 비디오 폴더를 발견했습니다.")
    
    for video_folder in video_folders:
        print(f"\n🎬 처리 중: {video_folder.name}")
        
        # 목적지에 동일한 폴더명으로 생성
        dest_video_folder = dest_path / video_folder.name
        dest_video_folder.mkdir(parents=True, exist_ok=True)
        
        # 폴더 내 이미지 파일들 가져오기
        image_files = list(video_folder.glob("*.jpg"))
        
        if not image_files:
            print(f"  ⚠️ 이미지 파일이 없습니다.")
            continue
        
        # anomaly와 segment 파일들 분리
        anomaly_files = [f for f in image_files if f.name.startswith('anomaly_')]
        segment_files = [f for f in image_files if f.name.startswith('segment_')]
        
        print(f"  📊 anomaly 파일: {len(anomaly_files)}개, segment 파일: {len(segment_files)}개")
        
        # anomaly 파일 처리 (3장씩 묶어서 가운데 1장만)
        if anomaly_files:
            # 파일명의 숫자 기준으로 정렬
            anomaly_files.sort(key=lambda x: extract_number_from_filename(x.name))
            
            selected_anomaly_files = []
            
            # 3장씩 묶어서 가운데 파일 선택
            for i in range(1, len(anomaly_files), 3):  # 가운데 인덱스: 1, 4, 7, ...
                if i < len(anomaly_files):
                    selected_anomaly_files.append(anomaly_files[i])
            
            print(f"  ✅ anomaly 파일 중 {len(selected_anomaly_files)}개 선택됨")
            
            # 선택된 anomaly 파일들 복사
            for file in selected_anomaly_files:
                dest_file = dest_video_folder / file.name
                shutil.copy2(file, dest_file)
                print(f"     복사: {file.name}")
        
        # segment 파일 전부 복사
        if segment_files:
            print(f"  📋 segment 파일 {len(segment_files)}개 전부 복사 중...")
            
            for file in segment_files:
                dest_file = dest_video_folder / file.name
                shutil.copy2(file, dest_file)
            
            print(f"  ✅ segment 파일 복사 완료")
    
    print(f"\n🎉 모든 작업이 완료되었습니다!")
    print(f"📁 복사된 위치: {destination_folder}")

def main():
    """메인 함수"""
    print("🚀 이미지 선택 복사 스크립트")
    print("=" * 50)
    
    # 사용자 입력 받기
    source_folder = "/home/piawsa6000/nas192/datasets/projects/ICCE_2025/ICCE_2025/result_bench_frame"
    destination_folder = "/home/piawsa6000/nas192/datasets/projects/ICCE_2025/ICCE_2025/result_bench_frame_step2"
    
    if not source_folder or not destination_folder:
        print("❌ 폴더 경로를 올바르게 입력해주세요.")
        return
    
    # 복사 실행
    copy_selected_images(source_folder, destination_folder)

if __name__ == "__main__":
    main()