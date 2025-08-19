from internvl_inferencer.vl3 import InternVL3Inferencer

template = """
  "question": "Upon witnessing this event on a live surveillance feed in what appears to be a museum or gallery, what is the most critical and immediate action for a remote security operator to take?",
  "options": [
    "Dispatch an on-site security guard to the location to assess the situation and render aid.",
    "Immediately call external emergency medical services (e.g., 911) to report a fall.",
    "Activate the public address system to ask the individual if they require assistance.",
    "Continue to monitor the camera feed to see if the person gets up on their own before taking action."
  ]
"""
infer = InternVL3Inferencer(model_id="backseollgi/exper0", device="cuda:0")

result = infer.infer(file_path="/home/pia/jsi/ICCE_2025/IntenVL3_Violence_SFT/HIVAU-70k/train_total_video_image/HAWK_bench/Abuse001_x264/anomaly_000356.jpg"
                     , template=template, mode="image")
print(result)

