from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai
import os
from assets.config import ENV_AUTH, PROMPT

class GeminiImageAnalyzer:
    def __init__(self, model_name: str = "gemini-2.5-flash", project: str = "gmail-361002", location: str = "us-central1"):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ENV_AUTH
        vertexai.init(project=project, location=location)
        self.model = GenerativeModel(model_name=model_name)
        self.model_name = model_name
    
    def analyze_image(self, image_path: str, custom_prompt: str = PROMPT) -> str:
        """
        이미지 분석을 수행합니다.
        
        Args:
            image_path (str): 분석할 이미지 경로
            custom_prompt (str): 커스텀 프롬프트 (선택사항)
        
        Returns:
            str: 분석 결과
        """
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            image_part = Part.from_data(data=image_bytes, mime_type="image/jpeg")
            prompt = custom_prompt
            response = self.model.generate_content([image_part, prompt])
            return response.text.strip()
        except Exception as e:
            return f"❌ gemini 오류 발생 ({self.model_name}): {str(e)}"
    
    def change_model(self, model_name: str):
        """모델을 변경합니다."""
        self.model = GenerativeModel(model_name=model_name)
        self.model_name = model_name