"""Shared helpers used by attack config modules."""

EVAL_QUERY = (
    "Describe the main object in this scene and explain its significance."
)

# Standard target strategies shared by physpatch / foa / mattack / advdiffvlm / advedm
STANDARD_TARGETS = {
    "stop_sign": {
        "description": "A stop sign",
        "reference_text": "A stop sign is visible",
        "target_image": "dataset/physpatch/images/target/stop_sign.png",
    },
    "plane": {
        "description": "An airplane",
        "reference_text": "An airplane is visible",
        "target_image": "dataset/physpatch/images/target/plane.jpg",
    },
    "pedestrian_crossing": {
        "description": "A pedestrian crossing sign",
        "reference_text": "A pedestrian crossing sign is visible",
        "target_image": "dataset/physpatch/images/target/pedestrian_crossing.png",
    },
    # extend here for new strategies
}
