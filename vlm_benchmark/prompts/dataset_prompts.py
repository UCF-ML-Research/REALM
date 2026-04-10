"""Per-dataset prompt formatting constants for VLM benchmark inference.

Paper alignment notes:
- DriveBench uses multi-choice questions with a driving system prompt
  (DRIVING_SYSTEM_PROMPT in system_prompts.py).
"""

# DriveBench multi-choice: letter only (GPT judge already skips MCQ;
# letter-only output maximizes logprob scoring coverage)
MULTICHOICE_SUFFIX = (
    "\nAnswer with ONLY the letter of the correct option (A, B, C, or D). "
    "Do not include any explanation."
)

# Short-answer prompt for open-ended questions
SHORT_ANSWER_SUFFIX = "\nAnswer in one or two sentences with specific details."

# Robo2VLM MCQ: Answer with letter only (matching official Robo2VLM)
# Official implementation uses letter-based answers (A/B/C/D/E)
ROBO2VLM_MCQ_SUFFIX = (
    "\nAnswer by selecting the letter (A, B, C, D, or E) only. "
    "ONLY output the correct option letter, i.e., A, B, C, D, E."
)

# Robo2VLM Goal Recognition: Special prompt for grid-based comparison tasks
# The image shows a 2x3 grid with 5 configurations labeled A-E in red text
ROBO2VLM_GOAL_RECOGNITION_SUFFIX = (
    "\n\nIMPORTANT: The image shows a GRID with 5 different robot configurations.\n"
    "Each panel in the grid has a red label (Configuration A, B, C, D, or E).\n\n"
    "⚠️  CRITICAL: Ignore the red configuration labels in the image!\n\n"
    "Your task:\n"
    "1. Read the question carefully to understand the goal\n"
    "2. Examine all 5 configurations in the grid\n"
    "3. Identify which configuration achieves the stated goal\n"
    "4. Look at the ANSWER CHOICES below (A, B, C, D, E)\n"
    "5. Each answer choice tells you which configuration it represents (e.g., 'A. Configuration C' means answer A shows Configuration C)\n"
    "6. Select the ANSWER LETTER (A/B/C/D/E) whose configuration matches the goal\n\n"
    "Example: If Configuration C achieves the goal, and you see 'A. Configuration C' in the choices, answer 'A' (not 'C').\n\n"
    "Answer with ONLY the answer choice letter (A, B, C, D, or E)."
)

# Perception VQA: guide model to list driving-relevant objects with action/motion context
PERCEPTION_VQA_SUFFIX = (
    "\nList the most important objects for driving (vehicles, pedestrians, traffic signs/lights, obstacles). "
    "For each object, state:\n"
    "- Its type, color, and position relative to the ego vehicle (e.g., 'to the front left')\n"
    "- Which camera view it appears in\n"
    "- Whether it is moving or stationary\n"
    "- What driving action the ego vehicle should consider in response "
    "(e.g., 'slow down', 'maintain speed', 'change lanes')\n"
    "Use natural language sentences. Focus on the 3-5 most safety-critical objects. "
    "Include their ID tags if available (e.g., <c1,CAM_FRONT,0.5,0.6>)."
)

# Perception MCQ: active trajectory reasoning to reduce "Going Ahead" over-prediction
MCQ_OBJECT_FOCUS_SUFFIX = (
    "\nFocus on the specific object referenced in the question. "
    "Examine the indicated camera view carefully.\n"
    "To determine the object's motion direction, follow these steps:\n"
    "1. Check if the object's wheels or body are angled (turning) or straight (going ahead)\n"
    "2. Check if the object is crossing or drifting across lane markings (turning) vs staying within lane (going ahead)\n"
    "3. Check multiple frames if available for position changes\n"
    "WARNING: Where the object SITS in the frame (left/right) does NOT tell you its MOTION direction. "
    "A car on the right side of the frame can be turning LEFT. "
    "Base your answer strictly on wheel angle, body orientation, and lane-crossing evidence."
)

# Prediction Yes/No: ensure answer starts with Yes/No for clean extraction
PREDICTION_YESNO_SUFFIX = (
    "\nAnswer with Yes or No first. "
    "Then briefly explain the object's actual category and appearance to justify your answer."
)

# Prediction VQA: match GT structure with correct camera names, state vocabulary, action calibration
PREDICTION_VQA_SUFFIX = (
    "\nYou MUST describe exactly THREE objects in priority order using this format:\n"
    "Firstly, notice that [color] [type] at [camera name]. "
    "The object is [moving/going ahead/stationary], so the ego vehicle should [action].\n"
    "Secondly, notice that [color] [type] at [camera name]. "
    "The object is [moving/going ahead/stationary], so the ego vehicle should [action].\n"
    "Thirdly, notice that [color] [type] at [camera name]. "
    "The object is [moving/going ahead/stationary], so the ego vehicle should [action].\n"
    "Camera names: use 'front camera', 'back camera', 'front left camera', "
    "'front right camera', 'back left camera', 'back right camera'.\n"
    "For object state, use: 'going ahead', 'turning left', 'turning right', or 'stationary'.\n"
    "For actions: prefer 'keep going ahead at the same speed' for non-threatening objects. "
    "Only recommend stopping or slowing if the object is directly in the ego vehicle's path.\n"
    "Describe ONLY objects you can clearly see. Do NOT hallucinate objects."
)

# Planning VQA suffixes (v4): split by question type for better alignment with GT format
# Type A (default ~60%): "What actions could the ego vehicle take...?"
PLANNING_ACTION_REASON_SUFFIX = (
    "\nAnswer in this exact format: "
    "'The action is to [ACTION]. The reason is [REASON], and the probability is [high/medium/low].'\n"
    "The probability refers to how confident you are that this is the CORRECT action to take "
    "(high = very confident this action is right, low = uncertain). "
    "Most routine driving situations have high probability.\n"
    "IMPORTANT: Most driving scenes are normal — the correct action is usually to keep driving "
    "(go straight, turn, change lanes, etc.). Only recommend stopping if a vehicle or obstacle "
    "is clearly visible in CAM_FRONT directly in the ego lane.\n"
    "Do NOT hallucinate obstacles or traffic lights that are not clearly visible in the images. "
    "Objects in CAM_BACK are behind the vehicle. Objects in side cameras may be in adjacent lanes."
)

# Type B (~20%): "What actions...can lead to a collision with...?"
PLANNING_COLLISION_SUFFIX = (
    "\nIdentify the driving maneuver that would cause a collision with the referenced object. "
    "Follow these steps:\n"
    "Step 1: Determine which CAMERA the object appears in from the question text.\n"
    "Step 2: Use this camera-to-action mapping:\n"
    "- CAM_BACK: The object is behind → 'Back up.' or 'Brake suddenly.'\n"
    "- CAM_FRONT: The object is ahead → 'Accelerate and go straight.' or a turn into the object's position\n"
    "- CAM_FRONT_LEFT: The object is front-left → 'Turn left.' (e.g., 'Sharp left turn.', 'Moderate left turn.')\n"
    "- CAM_FRONT_RIGHT: The object is front-right → 'Turn right.' (e.g., 'Sharp right turn.', 'Moderate right turn.')\n"
    "- CAM_BACK_LEFT: The object is rear-left → 'Back up and turn left.'\n"
    "- CAM_BACK_RIGHT: The object is rear-right → 'Back up and turn right.'\n"
    "Step 3: State the collision-causing action, then briefly explain why this maneuver "
    "would collide with the object, and state your confidence level (high, medium, or low).\n"
    "Answer format: 'The action is to [MANEUVER]. The reason is [WHY], "
    "and the probability is [high/medium/low].'"
)

# Type C (~20%): "What are safe actions to take...?"
PLANNING_SAFE_ACTIONS_SUFFIX = (
    "\nList the safe action(s) the ego vehicle can take in this scenario. "
    "In most scenes, it is safe to continue driving. Consider actions such as: "
    "keep going at the same speed, accelerate slightly, decelerate gradually, "
    "turn left, turn right, change lanes, or steer to offset.\n"
    "Do NOT assume red traffic lights or blocking obstacles exist unless they are "
    "clearly visible in the images. "
    "Include multiple safe options if applicable. "
    "For each action, briefly explain why it is safe."
)

# Backward-compatible alias
PLANNING_VQA_SUFFIX = PLANNING_ACTION_REASON_SUFFIX

# Behavior MCQ: direction + speed reasoning with anti-position-bias
BEHAVIOR_MCQ_FOCUS_SUFFIX = (
    "\nExamine road curvature, lane markings, and surrounding traffic flow "
    "to determine the ego vehicle's direction and speed. "
    "IMPORTANT: Consider ALL answer options carefully before selecting. "
    "Evaluate BOTH direction AND speed components independently. "
    "The correct answer must match both aspects precisely - do not default to any particular option."
)

# General fallback prompt when sample has no question
BEHAVIOR_FALLBACK_PROMPT = (
    "Predict the behavior of the ego vehicle."
)

# ============================================================================
# PhysPatch Dataset Prompts (AAAI 2026)
# ============================================================================
# Paper: https://arxiv.org/abs/2508.05167
# Original implementation prompt from vlm_response.py line 43

PHYSPATCH_PERCEPTION_PROMPT = (
    "Describe the main object in the scene that is most likely to influence "
    "the vehicle's next driving decision. You only need to describe the object "
    "in JSON format {'object': ,'describe:' }."
)

# Alternative planning prompt from paper (not used in main evaluation)
PHYSPATCH_PLANNING_PROMPT = (
    "The ego vehicle is in motion. What should it do next? "
    "(a) Brake; (b) Accelerate; (c) Turn Left; (d) Turn Right; (e) Go straight. "
    "Answer in JSON format."
)
