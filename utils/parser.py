import json
import re

def parse_json_response(response_text: str) -> dict:
    """
    문자열로 받은 JSON 응답을 파싱하여 딕셔너리로 반환합니다.
    
    다음 두 가지 케이스를 지원합니다:
    1. ```json { ... } ``` 형태
    2. { ... } 바로 시작하는 형태
    
    Args:
        response_text (str): 파싱할 응답 텍스트
    
    Returns:
        dict: 파싱된 JSON 데이터
        
    Raises:
        ValueError: JSON 파싱에 실패한 경우
    """
    if not response_text or not isinstance(response_text, str):
        raise ValueError("응답 텍스트가 비어있거나 문자열이 아닙니다.")
    
    # 공백 제거
    clean_text = response_text.strip()
    
    if not clean_text:
        raise ValueError("응답 텍스트가 비어있습니다.")
    
    try:
        # 케이스 1: ```json { ... } ``` 형태 처리
        if clean_text.startswith('```json') and clean_text.endswith('```'):
            # ```json과 ```를 제거
            json_content = clean_text[7:-3].strip()  # ```json (7글자) 제거, ``` (3글자) 제거
            return json.loads(json_content)
        
        # 케이스 2: ```json으로 시작하지만 ```로 끝나지 않는 경우
        elif clean_text.startswith('```json'):
            # ```json만 제거하고 마지막 ```가 있다면 제거
            json_content = clean_text[7:].strip()  # ```json (7글자) 제거
            if json_content.endswith('```'):
                json_content = json_content[:-3].strip()
            return json.loads(json_content)
        
        # 케이스 3: { ... } 바로 시작하는 형태
        elif clean_text.startswith('{') and clean_text.endswith('}'):
            return json.loads(clean_text)
        
        # 케이스 4: 텍스트 중간에 JSON이 포함된 경우 (정규표현식으로 추출)
        else:
            # { 로 시작해서 } 로 끝나는 JSON 패턴 찾기
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


def test_parser():
    """파서 테스트 함수"""
    
    # 테스트 케이스 1: ```json ``` 형태
    test1 = '''```json
{
  "question_type": "Multiple Choice",
  "question": "Test question?",
  "answer": "Test answer"
}
```'''
    
    # 테스트 케이스 2: 바로 { } 형태
    test2 = '''{
  "question_type": "Multiple Choice", 
  "question": "Test question?",
  "answer": "Test answer"
}'''
    
    # 테스트 케이스 3: 앞뒤에 다른 텍스트가 있는 경우
    test3 = '''Here is the response:
{
  "question_type": "Multiple Choice",
  "question": "Test question?", 
  "answer": "Test answer"
}
Hope this helps!'''

    test_cases = [
        ("케이스 1 (```json```)", test1),
        ("케이스 2 ({ } 직접)", test2), 
        ("케이스 3 (텍스트 중간)", test3)
    ]
    
    print("🧪 JSON 파서 테스트")
    print("=" * 50)
    
    for name, test_case in test_cases:
        try:
            result = parse_json_response(test_case)
            print(f"✅ {name}: 성공")
            print(f"   question_type: {result.get('question_type')}")
            print(f"   question: {result.get('question', '')[:50]}...")
            print()
        except Exception as e:
            print(f"❌ {name}: 실패 - {str(e)}")
            print()

if __name__ == "__main__":
    test_parser()