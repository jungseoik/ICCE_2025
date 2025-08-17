import json
import os

def parse_anomaly_jsonl(input_file, output_file):
    """
    JSONL 파일을 파싱하여 비디오 경로와 normal/anomal 라벨만 추출
    
    Args:
        input_file (str): 입력 JSONL 파일 경로
        output_file (str): 출력 JSONL 파일 경로
    """
    parsed_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # JSON 파싱
                data = json.loads(line.strip())
                
                # type이 "event"이고 task가 "judgement"인 경우만 처리
                if data.get("type") == "event" and data.get("task") == "judgement":
                    video_path = data.get("video", "")
                    
                    # conversations에서 gpt 응답 찾기
                    conversations = data.get("conversations", [])
                    gpt_response = ""
                    
                    for conv in conversations:
                        if conv.get("from") == "gpt":
                            gpt_response = conv.get("value", "")
                            break
                    
                    # 라벨 결정 (대소문자 구분 없이)
                    gpt_response_lower = gpt_response.lower()
                    
                    if "no anomaly" in gpt_response_lower:
                        label = "normal"
                    else:
                        # "no anomaly"가 없으면 anomaly로 간주
                        label = "anomal"
                    
                    # 결과 저장
                    parsed_entry = {
                        "video": video_path,
                        "label": label
                    }
                    
                    parsed_data.append(parsed_entry)
                    
            except json.JSONDecodeError:
                print(f"Warning: Line {line_num}에서 JSON 파싱 오류 발생")
                continue
            except Exception as e:
                print(f"Warning: Line {line_num}에서 오류 발생: {e}")
                continue
    
    # 결과를 JSONL 파일로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in parsed_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"파싱 완료: {len(parsed_data)}개 항목이 {output_file}에 저장되었습니다.")
    
    # 통계 출력
    normal_count = sum(1 for item in parsed_data if item['label'] == 'normal')
    anomal_count = sum(1 for item in parsed_data if item['label'] == 'anomal')
    
    print(f"통계:")
    print(f"  - Normal: {normal_count}개")
    print(f"  - Anomal: {anomal_count}개")
    
    return parsed_data

def check_parsing_results(data):
    """파싱 결과를 확인하는 함수"""
    print("\n샘플 결과:")
    for i, item in enumerate(data[:5]):  # 처음 5개만 출력
        print(f"{i+1}. Video: {item['video']}, Label: {item['label']}")

# 사용 예시
if __name__ == "__main__":
    input_file = "/home/pia/jsi/ICCE_2025/ucf-crime/merge_instruction_train_final.jsonl"  # 입력 파일명을 여기에 입력
    output_file = "result_ab_nor_extract.jsonl"  # 출력 파일명
    
    # 파일이 존재하는지 확인
    if not os.path.exists(input_file):
        print(f"오류: {input_file} 파일을 찾을 수 없습니다.")
        print("파일명을 확인하거나 파일을 현재 디렉토리에 배치해주세요.")
    else:
        # 파싱 실행
        parsed_data = parse_anomaly_jsonl(input_file, output_file)
        
        # 결과 확인
        check_parsing_results(parsed_data)
