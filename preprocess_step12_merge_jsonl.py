import json

def merge_jsonl_with_continuous_ids(file1_path, file2_path, output_path):
    """
    두 개의 JSONL 파일을 병합하면서 두 번째 파일의 ID를 첫 번째 파일에 이어지도록 조정합니다.

    Args:
        file1_path (str): 첫 번째 JSONL 파일 경로.
        file2_path (str): 병합할 두 번째 JSONL 파일 경로.
        output_path (str): 병합된 결과가 저장될 파일 경로.
    """
    max_id = 0
    
    # 1. 첫 번째 파일에서 최대 ID 찾기 및 새로운 파일에 내용 복사
    with open(file1_path, 'r', encoding='utf-8') as f1, open(output_path, 'w', encoding='utf-8') as out_f:
        for line in f1:
            try:
                data = json.loads(line)
                # 현재 라인의 id가 max_id보다 크면 업데이트
                if 'id' in data and data['id'] > max_id:
                    max_id = data['id']
                out_f.write(line)
            except json.JSONDecodeError:
                print(f"Warning: '{file1_path}' 파일의 다음 라인을 파싱할 수 없습니다: {line.strip()}")

    print(f"첫 번째 파일의 최대 ID: {max_id}")

    # 2. 두 번째 파일을 읽어오면서 ID를 새로 부여하여 병합 파일에 추가
    next_id = max_id + 1
    with open(file2_path, 'r', encoding='utf-8') as f2, open(output_path, 'a', encoding='utf-8') as out_f:
        for line in f2:
            try:
                data = json.loads(line)
                # 새로운 ID 부여
                data['id'] = next_id
                # JSON 형식의 문자열로 변환하여 파일에 쓰기
                out_f.write(json.dumps(data, ensure_ascii=False) + '\n')
                next_id += 1
            except json.JSONDecodeError:
                print(f"Warning: '{file2_path}' 파일의 다음 라인을 파싱할 수 없습니다: {line.strip()}")

    print(f"병합 완료! 총 {next_id - 1}개의 데이터가 '{output_path}' 파일에 저장되었습니다.")

# --- 코드 실행 ---
# 사용자의 파일 경로를 지정해주세요.
file1 = '/home/pia/jsi/ICCE_2025/IntenVL3_Violence_SFT/HIVAU-70k/exeperiments/exp2_hivau_score_clsvd_clsdt.jsonl'
file2 = 'result_video_converted.jsonl'
merged_file = 'merged_file.jsonl'

# 함수 호출하여 파일 병합 실행
merge_jsonl_with_continuous_ids(file1, file2, merged_file)