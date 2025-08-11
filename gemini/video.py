from google import genai
from google import genai

client = genai.Client(api_key="AIzaSyCdYdL7PaY_Lpetd8-tBEIm-p-X-pU-G8E")

myfile = client.files.upload(file="/home/piawsa6000/nas192/datasets/projects/ICCE_2025/ucf_crime/ucf-crime/videos/train/Abuse011_x264.mp4")

response = client.models.generate_content(
    model="gemini-2.5-flash", contents=[myfile, "Summarize this video. Then create a quiz with an answer key based on the information in this video."]
)

print(response.text)