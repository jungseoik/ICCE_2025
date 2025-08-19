import json

file_path = "/home/pia/jsi/ICCE_2025/IntenVL3_Violence_SFT/HIVAU-70k/exeperiments/exp5_hivau_score_clsvd_clsdt_video.jsonl"

with open(file_path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:  # 빈 줄 건너뛰기
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f"에러 발견!")
            print(f"라인 번호: {line_num}")
            print(f"에러 내용: {e}")
            print(f"문제가 있는 라인: {line}")
            print(f"에러 위치: {e.pos}번째 문자")
            if len(line) > e.pos:
                print(f"에러 지점 주변: ...{line[max(0, e.pos-20):e.pos+20]}...")
            break
    else:
        print("JSONL 파일이 정상입니다!")