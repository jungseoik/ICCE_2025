from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai
import os
import mimetypes
from assets.config import ENV_AUTH, PROMPT , PROMPT_VIDEO

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
    
    def analyze_video(
        self,
        video_path: str,
        custom_prompt: str = PROMPT_VIDEO,
        mime_type: str  = "video/mp4",
    ) -> str:
        """
        비디오 분석
        
        Args:
            video_path (str): 분석할 비디오 경로
            custom_prompt (str): 커스텀 프롬프트 (선택, 기본: PROMPT_VIDEO)
            mime_type (str | None): 강제 MIME 타입 지정 (미지정 시 확장자로 추정, 기본값 video/mp4)
        Returns:
            str: 분석 결과 텍스트
        """
        try:
            with open(video_path, "rb") as f:
                video_bytes = f.read()

            if mime_type is None:
                guessed_mime, _ = mimetypes.guess_type(video_path)
                mime_type = guessed_mime or "video/mp4"

            video_part = Part.from_data(data=video_bytes, mime_type=mime_type)
            response = self.model.generate_content([video_part, custom_prompt])
            return (response.text or "").strip()
        except Exception as e:
            return f"❌ gemini 오류 발생 ({self.model_name}): {str(e)}"
        
    def change_model(self, model_name: str):
        """모델을 변경합니다."""
        self.model = GenerativeModel(model_name=model_name)
        self.model_name = model_name