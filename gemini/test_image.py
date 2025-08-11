from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "gmail-361002-cbcf95afec4a.json"
vertexai.init(project="gmail-361002", location="us-central1")

model = GenerativeModel(model_name="gemini-1.5-pro-002")

image_path = "289.jpg"
with open(image_path, "rb") as f:
    image_bytes = f.read()

image_part = Part.from_data(data=image_bytes, mime_type="image/jpeg")
prompt = "이 이미지를 설명해줘."
response = model.generate_content([image_part, prompt])
print(response.text)
