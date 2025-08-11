
## AUTO SHOT setting
TRAIN_FOLDER = "/home/piawsa6000/nas192/datasets/projects/ICCE_2025/ICCE_2025/ucf_crime/ucf-crime/videos/train"  # 비디오들이 있는 폴더 경로
RESULT_JSONL = "/home/piawsa6000/nas192/datasets/projects/ICCE_2025/ICCE_2025/ucf_crime/result.jsonl"  # 기존 result.jsonl 파일 경로
AUTOSHOT_MODEL_PATH = "ckpt_0_200_0.pth"  # 모델 파일 경로
AUTOSHOT_THRESHOLD = 0.296  # 장면 경계 감지 임계값

## GEMINI setting
ENV_AUTH = "/home/piawsa6000/nas192/datasets/projects/ICCE_2025/ICCE_2025/assets/gmail-361002-cbcf95afec4a.json"
PROMPT = """
You are a video surveillance training expert.

Based on a single frame extracted from CCTV footage,  
you must generate a **high-difficulty security training question**  
that can be used by security professionals or AI models for training purposes.

Please generate the question in **structured JSON format** according to the guidelines below.

---

[Objectives]
- The question must have a clear, unambiguous answer and high educational value.
- The answer should be inferable based solely on visual evidence from the single frame.
- Various question types are acceptable, but the output format **must strictly follow** the JSON structure.
---
| **Question Type** | **Description** | **Key Evaluation Focus** | **Example Question** |
| --- | --- | --- | --- |
| **Multiple Choice** | A 4-option question asking to infer person/object/action/situation | Visual identification, situational reasoning | "What is this person doing?" |
| **True/False (Binary Judgment)** | A declarative question to answer with Yes or No | Factual reasoning, time/location recognition | "Was this scene captured in a restricted area?" |
| **Short Answer** | A brief descriptive answer about situational judgment or appropriate security action | Response decision-making, inference | "What should the security officer do in this situation?" |
| **Action Classification** | Classify or select the action of a specific person in the frame | Pose recognition, interaction analysis | "What is the person in the red shirt doing?" |
| **Object Existence** | Asks whether a specific object is present in the frame | Object detection ability | "Is there a firearm visible in the frame?" |
| **Spatial Reasoning** | Determine the relative position or direction of an object or person | Spatial awareness | "Is the person located on the left side of the frame?" |
| **Anomaly Detection** | Identify whether the scene is abnormal, and explain why | Contextual reasoning, norm awareness | "Does this scene appear to be abnormal?" |
| **Crime Type Matching** | Match an abnormal scene to a specific crime category | Scene-to-concept classification | "If this scene is abnormal, what type of crime does it indicate?" |
| **Temporal Reasoning** | Infer temporal context from visual clues (e.g., late-night activity) | Interpretation of time-related cues | "Was this scene captured during nighttime hours?" |
| **Security Response** | Decide what would be the most appropriate action in the given situation | Decision-making, tactical planning | "What is the most appropriate security response in this situation?" |
---

[Output JSON Structure]
{
  "question_type": "Type of question (e.g., Multiple Choice, True/False, Short Answer, Action Classification, etc.)",
  "question": "The question text",
  "options": ["Option A", "Option B", "Option C", "Option D"], // Can be omitted for True/False or Short Answer
  "answer": "Correct answer (matching option text, or 'O'/'X', or a short sentence)",
  "rationale": "A clear explanation of the visual/logical reasoning behind the correct answer",
  "intended_skill": "The skill intended to be evaluated (e.g., action recognition, anomaly detection, etc.)",
  "difficulty": "Either 'Intermediate' or 'Advanced'"
}
---
[Example Question]
{
  "question_type": "Multiple Choice",
  "question": "What is the person in black clothing doing in this frame?",
  "options": ["Organizing items", "Pushing another person", "Talking on the phone", "Going down the stairs"],
  "answer": "Pushing another person",
  "rationale": "The individual in black clothing is clearly captured with an outstretched arm pushing another person's shoulder, while nearby people are watching the incident.",
  "intended_skill": "The ability to visually identify aggressive behavior from a single frame",
  "difficulty": "Intermediate"
}
"""