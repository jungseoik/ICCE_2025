from gemini.gemini_api import GeminiImageAnalyzer
import os
import json
import re
from pathlib import Path
import mimetypes
from typing import Dict, Tuple, Optional
from assets.config import PROMPT_VIDEO  # 기본 비디오 프롬프트


def parse_json_response(response_text: str) -> dict:
    """
    문자열로 받은 JSON 응답을 파싱하여 딕셔너리로 반환합니다.
    (이전 이미지 파이프라인 로직 그대로 사용)
    """
    if not response_text or not isinstance(response_text, str):
        raise ValueError("응답 텍스트가 비어있거나 문자열이 아닙니다.")
    
    clean_text = response_text.strip()
    if not clean_text:
        raise ValueError("응답 텍스트가 비어있습니다.")

    try:
        # ```json ... ``` 케이스
        if clean_text.startswith("```json") and clean_text.endswith("```"):
            return json.loads(clean_text[7:-3].strip())
        elif clean_text.startswith("```json"):
            body = clean_text[7:].strip()
            if body.endswith("```"):
                body = body[:-3].strip()
            return json.loads(body)
        # { ... } 바로 시작/끝
        elif clean_text.startswith("{") and clean_text.endswith("}"):
            return json.loads(clean_text)
        # 텍스트 중간에 JSON
        else:
            match = re.search(r"\{.*\}", clean_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError("JSON 형태를 찾을 수 없습니다.")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 파싱 오류: {str(e)}")
    except Exception as e:
        raise ValueError(f"예상치 못한 오류: {str(e)}")


def load_labels_jsonl(jsonl_path: str) -> Dict[str, str]:
    """
    JSONL에서 비디오 파일명 -> 라벨 매핑을 로드합니다.
    {"video": "ucf-crime/events/train/Abuse001_x264_E0.mp4", "label": "anomal"}
    처럼 경로가 포함돼도 파일명만 키로 사용합니다.
    """
    labels = {}
    p = Path(jsonl_path)
    if not p.exists():
        raise FileNotFoundError(f"라벨 JSONL 파일이 없습니다: {jsonl_path}")

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                video_field = obj.get("video", "")
                label = obj.get("label", "")
                if not video_field:
                    continue
                fname = Path(video_field).name
                labels[fname] = label
            except Exception:
                # 깨진 라인 스킵
                continue
    return labels


def normalize_label(label: Optional[str]) -> str:
    """
    라벨 표기를 정규화합니다.
    """
    if not label:
        return "unknown"
    l = label.strip().lower()
    if l in {"anomal", "anomaly", "abnormal", "anormal", "anomalous"}:
        return "anomal"
    if l in {"normal", "norm"}:
        return "normal"
    return l


def choose_api_by_label(
    label: str,
    api_flash: GeminiImageAnalyzer,
    api_pro: GeminiImageAnalyzer,
) -> Tuple[GeminiImageAnalyzer, str]:
    """
    기존 이미지 파이프라인과 유사하게 라벨에 따라 모델 선택:
    - anomal 계열 → pro
    - normal → flash
    - 그 외/unknown → pro (보수적으로 고성능 모델)
    """
    n = normalize_label(label)
    if n == "normal":
        return api_flash, "flash"
    else:
        return api_pro, "pro"


def process_single_video(
    video_file: Path,
    output_folder: Path,
    api: GeminiImageAnalyzer,
    label: str,
    custom_prompt: Optional[str] = None,
) -> bool:
    """
    단일 비디오를 분석하고 JSON으로 저장합니다.
    """
    try:
        print(f"    🎥 분석 중: {video_file.name} (label: {label}, model: {api.model_name})")

        prompt = custom_prompt or PROMPT_VIDEO
        result_text = api.analyze_video(video_path=str(video_file), custom_prompt=prompt)

        if not result_text:
            print(f"    ❌ 분석 결과가 비어있습니다: {video_file.name}")
            return False
        
        # JSON 파싱 시도
        try:
            json_data = parse_json_response(result_text)
        except ValueError as e:
            print(f"    ⚠️ JSON 파싱 실패, raw 저장: {video_file.name} - {str(e)}")
            json_data = {
                "error": "JSON 파싱 실패",
                "raw_response": result_text,
            }

        # 메타데이터
        json_data["video_file"] = video_file.name
        json_data["label_from_jsonl"] = normalize_label(label)
        json_data["model"] = getattr(api, "model_name", "unknown")

        # 저장
        json_file_path = output_folder / (video_file.stem + ".json")
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        print(f"    ✅ 저장 완료: {json_file_path.name}")
        return True

    except Exception as e:
        print(f"    ❌ 오류 발생: {video_file.name} - {str(e)}")
        return False


def process_videos_in_folder(
    source_folder: str,
    labels_jsonl_path: str,
    output_folder: Optional[str] = None,
    video_exts=(".mp4", ".mov", ".mkv", ".avi"),
    recursive: bool = True,
):
    """
    폴더 내 비디오들을 분석하고 JSON으로 저장합니다.
    - labels_jsonl_path에서 라벨을 불러와 비디오 파일명과 매칭합니다.
    - 라벨별로 모델(Flash/Pro)을 선택해 분석합니다.
    """
    source_path = Path(source_folder)
    if not source_path.exists():
        raise FileNotFoundError(f"소스 폴더가 없습니다: {source_folder}")

    # 출력 폴더
    output_path = Path(output_folder) if output_folder else (source_path.parent / f"{source_path.name}_json")
    output_path.mkdir(parents=True, exist_ok=True)

    # 모델 준비
    flash_api = GeminiImageAnalyzer("gemini-2.5-flash")  # normal용
    pro_api = GeminiImageAnalyzer("gemini-2.5-pro")      # anomal/unknown용

    # 라벨 로드
    labels_map = load_labels_jsonl(labels_jsonl_path)
    print(f"🚀 비디오 분석 시작")
    print(f"📂 소스 폴더: {source_folder}")
    print(f"🧾 라벨 파일: {labels_jsonl_path} (총 {len(labels_map)}개 항목)")
    print(f"💾 출력 폴더: {output_path}")
    print("=" * 60)

    # 비디오 수집
    pattern = "**/*" if recursive else "*"
    video_files = [p for p in source_path.glob(pattern) if p.is_file() and p.suffix.lower() in video_exts]
    if not video_files:
        print("❌ 비디오 파일을 찾을 수 없습니다.")
        return

    total, ok, fail, skipped = 0, 0, 0, 0
    for vf in sorted(video_files):
        total += 1
        label = labels_map.get(vf.name, "unknown")
        if label == "unknown":
            print(f"  ⚠️ 라벨 없음(unknown): {vf.name} → 처리는 계속합니다.")
        api, chosen = choose_api_by_label(label, flash_api, pro_api)

        out_dir = output_path  # 결과 저장 폴더
        json_file_path = out_dir / (vf.stem + ".json")

        # ✅ 이미 JSON이 있으면 건너뛰기
        if json_file_path.exists():
            print(f"  ⏩ 이미 처리됨: {vf.name} → 건너뜀")
            skipped += 1
            continue

        success = process_single_video(vf, out_dir, api, label)
        if success:
            ok += 1
        else:
            fail += 1

    # for vf in sorted(video_files):
    #     total += 1
    #     label = labels_map.get(vf.name, "unknown")
    #     if label == "unknown":
    #         print(f"  ⚠️ 라벨 없음(unknown): {vf.name} → 처리는 계속합니다.")
    #     api, chosen = choose_api_by_label(label, flash_api, pro_api)

    #     # 비디오별 하위 폴더(선택): 동일 파일명이 여러 경로에 있을 수도 있어 안전하게 구분하고 싶다면 사용
    #     # out_dir = output_path / vf.stem
    #     # out_dir.mkdir(parents=True, exist_ok=True)
    #     out_dir = output_path  # 단일 폴더에 저장

    #     success = process_single_video(vf, out_dir, api, label)
    #     if success:
    #         ok += 1
    #     else:
    #         fail += 1

    print(f"\n🎉 처리 완료!")
    print(f"📊 총 처리: {total}개")
    print(f"✅ 성공: {ok}개")
    print(f"❌ 실패: {fail}개")
    print(f"💾 결과 저장 위치: {output_path}")


def main():
    """메인"""
    print("🎯 비디오 분석 및 JSON 생성 스크립트")
    print("=" * 50)

    # ▶️ 사용자 환경에 맞게 경로만 바꿔주세요
    source_folder = "IntenVL3_Violence_SFT/HIVAU-70k/train_total_video_image/ucf-crime/events/train"  # 비디오들이 있는 루트 폴더
    labels_jsonl_path = "result_ab_nor_extract.jsonl"
    output_folder = "IntenVL3_Violence_SFT/HIVAU-70k/train_total_video_image/HAWK_bench_video_json"  # 생략하면 기본: <소스폴더명>_json

    process_videos_in_folder(
        source_folder=source_folder,
        labels_jsonl_path=labels_jsonl_path,
        output_folder=output_folder,
        recursive=True,  # 하위 디렉토리까지 모두 탐색
    )


if __name__ == "__main__":
    main()
