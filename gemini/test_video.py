from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "gmail-361002-cbcf95afec4a.json"
vertexai.init(project="gmail-361002", location="us-central1")

# model = GenerativeModel(model_name="gemini-1.5-pro-002")
model = GenerativeModel(model_name="gemini-2.5-pro-preview-03-25")



video_path = "109-5_cam02_swoon02_place01_night_summer_5614_5734.mp4"
video_path = "/home/piawsa6000/nas192/datasets/projects/ICCE_2025/ucf_crime/ucf-crime/videos/train/Abuse011_x264.mp4"
with open(video_path, "rb") as f:
    video_bytes = f.read()

video_part = Part.from_data(data=video_bytes, mime_type="video/mp4")

prompt = "이상상황이 발생하는 구간 말해봐"

response = model.generate_content([video_part, prompt])
print(response.text)
