from gemini.gemini_api import GeminiImageAnalyzer
import os
import json
from pathlib import Path

def parse_json_response(response_text: str) -> dict:
    """
    문자열로 받은 JSON 응답을 파싱하여 딕셔너리로 반환합니다.
    
    Args:
        response_text (str): 파싱할 응답 텍스트
    
    Returns:
        dict: 파싱된 JSON 데이터
    """
    import re
    
    if not response_text or not isinstance(response_text, str):
        raise ValueError("응답 텍스트가 비어있거나 문자열이 아닙니다.")
    
    clean_text = response_text.strip()
    
    if not clean_text:
        raise ValueError("응답 텍스트가 비어있습니다.")
    
    try:
        # 케이스 1: ```json { ... } ``` 형태 처리
        if clean_text.startswith('```json') and clean_text.endswith('```'):
            json_content = clean_text[7:-3].strip()
            return json.loads(json_content)
        
        # 케이스 2: ```json으로 시작하지만 ```로 끝나지 않는 경우
        elif clean_text.startswith('```json'):
            json_content = clean_text[7:].strip()
            if json_content.endswith('```'):
                json_content = json_content[:-3].strip()
            return json.loads(json_content)
        
        # 케이스 3: { ... } 바로 시작하는 형태
        elif clean_text.startswith('{') and clean_text.endswith('}'):
            return json.loads(clean_text)
        
        # 케이스 4: 텍스트 중간에 JSON이 포함된 경우
        else:
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, clean_text, re.DOTALL)
            
            if match:
                json_content = match.group(0)
                return json.loads(json_content)
            else:
                raise ValueError("JSON 형태를 찾을 수 없습니다.")
                
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 파싱 오류: {str(e)}")
    except Exception as e:
        raise ValueError(f"예상치 못한 오류: {str(e)}")


def process_images_in_folder(source_folder: str, output_folder: str = None):
    """
    폴더 내 이미지들을 분석하고 JSON으로 저장합니다.
    
    Args:
        source_folder (str): result_bench_frame_step2 폴더 경로
        output_folder (str): JSON 파일을 저장할 폴더 (None이면 소스 폴더와 동일한 구조)
    """
    source_path = Path(source_folder)
    
    # 출력 폴더 설정
    if output_folder is None:
        output_path = source_path.parent / f"{source_path.name}_json"
    else:
        output_path = Path(output_folder)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Gemini API 인스턴스 생성
    flash_api = GeminiImageAnalyzer("gemini-2.5-flash")  # segment용
    pro_api = GeminiImageAnalyzer("gemini-2.5-pro")     # anomaly용
    
    print(f"🚀 이미지 분석 시작")
    print(f"📂 소스 폴더: {source_folder}")
    print(f"💾 출력 폴더: {output_path}")
    print("=" * 60)
    
    # 비디오 폴더들 탐색
    video_folders = [folder for folder in source_path.iterdir() if folder.is_dir()]
    
    if not video_folders:
        print("❌ 비디오 폴더를 찾을 수 없습니다.")
        return
    
    total_processed = 0
    total_success = 0
    total_failed = 0
    
    for video_folder in video_folders:
        print(f"\n🎬 처리 중: {video_folder.name}")
        
        # 출력 폴더에 동일한 비디오 폴더 생성
        output_video_folder = output_path / video_folder.name
        output_video_folder.mkdir(parents=True, exist_ok=True)
        
        # 이미지 파일들 가져오기
        image_files = list(video_folder.glob("*.jpg"))
        
        if not image_files:
            print(f"  ⚠️ 이미지 파일이 없습니다.")
            continue
        
        # anomaly와 segment 파일들 분리
        anomaly_files = [f for f in image_files if f.name.startswith('anomaly_')]
        segment_files = [f for f in image_files if f.name.startswith('segment_')]
        
        print(f"  📊 anomaly: {len(anomaly_files)}개, segment: {len(segment_files)}개")
        
        # anomaly 파일들 처리 (gemini-2.5-pro)
        for image_file in anomaly_files:
            success = process_single_image(
                image_file, 
                output_video_folder, 
                pro_api, 
                "anomaly"
            )
            total_processed += 1
            if success:
                total_success += 1
            else:
                total_failed += 1
        
        # segment 파일들 처리 (gemini-2.5-flash)
        for image_file in segment_files:
            success = process_single_image(
                image_file, 
                output_video_folder, 
                flash_api, 
                "segment"
            )
            total_processed += 1
            if success:
                total_success += 1
            else:
                total_failed += 1
    
    print(f"\n🎉 처리 완료!")
    print(f"📊 총 처리: {total_processed}개")
    print(f"✅ 성공: {total_success}개") 
    print(f"❌ 실패: {total_failed}개")
    print(f"💾 결과 저장 위치: {output_path}")


def process_single_image(image_file: Path, output_folder: Path, api: GeminiImageAnalyzer, image_type: str) -> bool:
    """
    단일 이미지를 처리하고 JSON으로 저장합니다.
    
    Args:
        image_file (Path): 처리할 이미지 파일 경로
        output_folder (Path): JSON을 저장할 폴더
        api (GeminiImageAnalyzer): 사용할 Gemini API 인스턴스
        image_type (str): 이미지 타입 ("anomaly" 또는 "segment")
    
    Returns:
        bool: 성공 여부
    """
    try:
        print(f"    🔍 분석 중: {image_file.name} ({image_type})")
        
        # Gemini API로 이미지 분석
        result = api.analyze_image(image_path=str(image_file))
        
        if not result:
            print(f"    ❌ 분석 결과가 비어있습니다: {image_file.name}")
            return False
        
        # JSON 파싱
        try:
            json_data = parse_json_response(result)
        except ValueError as e:
            print(f"    ❌ JSON 파싱 실패: {image_file.name} - {str(e)}")
            # 파싱 실패 시 원본 텍스트를 저장
            json_data = {
                "error": "JSON 파싱 실패",
                "raw_response": result,
                "image_type": image_type
            }
        
        # 메타데이터 추가
        json_data["image_file"] = image_file.name
        json_data["image_type"] = image_type
        json_data["model"] = api.model_name if hasattr(api, 'model_name') else "unknown"
        
        # JSON 파일로 저장 (이미지 파일명과 동일한 이름)
        json_filename = image_file.stem + ".json"  # .jpg -> .json
        json_file_path = output_folder / json_filename
        
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"    ✅ 저장 완료: {json_filename}")
        return True
        
    except Exception as e:
        print(f"    ❌ 오류 발생: {image_file.name} - {str(e)}")
        return False


def main():
    """메인 함수"""
    print("🎯 이미지 분석 및 JSON 생성 스크립트")
    print("=" * 50)
    
    # 사용자 입력 받기
    source_folder = "/home/piawsa6000/nas192/datasets/projects/ICCE_2025/ICCE_2025/result_bench_frame_step2"
    # 출력 폴더 옵션
    use_custom_output = "/home/piawsa6000/nas192/datasets/projects/ICCE_2025/ICCE_2025/result_bench_frame_step2_json"
    process_images_in_folder(source_folder, use_custom_output)


if __name__ == "__main__":
    main()
