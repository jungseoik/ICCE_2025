import json
import os

def convert_jsonl(input_file, output_file):
    """
    JSONL 파일을 읽어서 형식을 변환하고 저장하는 함수
    
    Args:
        input_file: 입력 JSONL 파일 경로
        output_file: 출력 JSONL 파일 경로
    """
    
    # 새로운 프롬프트 (영어 버전)
    new_prompt = """<video>
Evaluate the risk level of this video on a scale from 0.0 to 1.0.

Scoring Guidelines:
- 0.0-0.2: Completely normal daily activities (peaceful walking, conversation, shopping, etc.)
- 0.3-0.4: Slight anomalies (rapid movement, crowding, minor disputes)
- 0.5-0.6: Situations requiring attention (suspicious behavior, potential danger signals)
- 0.7-0.8: Clear dangerous situations (physical collisions, theft, threatening behavior)
- 0.9-1.0: Serious emergency situations (violence, clear criminal acts)

Examples:
- People calmly walking in a mall → {"score": 0.0}
- Normal traffic flow on street → {"score": 0.0}
- Customers shopping peacefully → {"score": 0.0}
- Someone running in a hallway → {"score": 0.3}
- Heated verbal argument → {"score": 0.4}
- Person loitering suspiciously near ATM → {"score": 0.5}
- Group gathering aggressively → {"score": 0.6}
- Physical pushing and shoving → {"score": 0.7}
- Robbery in progress → {"score": 0.8}
- Violent assault occurring → {"score": 0.9}
- Armed attack or shooting → {"score": 1.0}

Important: For this binary classification task:
- If the video shows ANY abnormal, dangerous, or criminal activity → Return {"score": 1.0}
- If the video shows ONLY normal, safe, everyday activities → Return {"score": 0.0}

Respond in JSON format as {"score": X.X}, where X.X represents a value between 0.0 and 1.0
"""
    
    converted_data = []
    
    try:
        # 입력 파일 읽기
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # 빈 줄 무시
                    try:
                        # JSON 파싱
                        data = json.loads(line)
                        
                        # task가 "clsvd"인 경우만 처리
                        if data.get('task') != 'clsvd':
                            continue
                        
                        # conversations 부분 수정
                        if 'conversations' in data and len(data['conversations']) >= 2:
                            # human 파트 수정 (프롬프트 교체)
                            data['conversations'][0]['value'] = new_prompt
                            
                            # gpt 파트 수정 (category를 score로 변환)
                            gpt_response = data['conversations'][1]['value']
                            
                            # 대소문자 구분 없이 처리하기 위해 소문자로 변환
                            gpt_response_lower = gpt_response.lower()
                            
                            # normal/abnormal을 score로 변환 (대소문자 무관)
                            if '"category": "normal"' in gpt_response_lower or '"category":"normal"' in gpt_response_lower:
                                new_score = 0.0
                            elif '"category": "abnormal"' in gpt_response_lower or '"category":"abnormal"' in gpt_response_lower:
                                new_score = 1.0
                            else:
                                # 예외 처리: 기본값 설정
                                print(f"Warning: Unexpected category format in ID {data.get('id', 'unknown')}: {gpt_response}")
                                new_score = 0.0
                            
                            # gpt 응답 업데이트
                            data['conversations'][1]['value'] = json.dumps({"score": new_score})
                        
                        converted_data.append(data)
                        
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON line: {e}")
                        print(f"Problematic line: {line[:100]}...")  # 처음 100자만 출력
                        continue
        
        # 결과를 새 파일에 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in converted_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✅ Successfully converted {len(converted_data)} entries")
        print(f"📁 Saved to: {output_file}")
        
        # 변환 예시 출력
        if converted_data:
            print("\n📋 Conversion Example:")
            print("First entry after conversion:")
            print(json.dumps(converted_data[0], indent=2, ensure_ascii=False))
        
        return converted_data
        
    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_file}' not found")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def main():
    """메인 실행 함수"""
    
    # 파일 경로 설정
    input_file = "result_train_ucf_clvd.jsonl"  # 입력 파일 경로를 실제 파일명으로 변경하세요
    output_file = "result_train_score_vad.jsonl"
    
    print("🔄 Starting JSONL conversion...")
    print(f"📂 Input file: {input_file}")
    print(f"📂 Output file: {output_file}")
    print(f"🎯 Processing only entries with task='clsvd'")
    print("-" * 50)
    
    # 변환 실행
    result = convert_jsonl(input_file, output_file)
    
    if result:
        print("-" * 50)
        print("✨ Conversion completed successfully!")
        
        # 통계 출력
        normal_count = sum(1 for item in result 
                          if item['conversations'][1]['value'] == '{"score": 0.0}')
        abnormal_count = sum(1 for item in result 
                            if item['conversations'][1]['value'] == '{"score": 1.0}')
        
        print(f"\n📊 Statistics:")
        print(f"  - Total 'clsvd' entries processed: {len(result)}")
        print(f"  - Normal (score 0.0): {normal_count}")
        print(f"  - Abnormal (score 1.0): {abnormal_count}")
    else:
        print("❌ Conversion failed!")

if __name__ == "__main__":
    main()