import os
from dataclasses import dataclass

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

torch.use_deterministic_algorithms(False)

import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel
from autodistill.helpers import load_image
from groundingdino.util.inference import Model

from autodistill_grounding_dino.helpers import combine_detections, load_grounding_dino

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class GroundingDINO(DetectionBaseModel):
    ontology: CaptionOntology
    grounding_dino_model: Model
    box_threshold: float
    text_threshold: float

    def __init__(self, box_threshold=0.35, text_threshold=0.25):
        self.grounding_dino_model = load_grounding_dino()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.ontology = CaptionOntology({"person": "person"})

    def predict(self, input: str, prompts) -> sv.Detections:
        image = load_image(input, return_format="cv2")
        ontology = CaptionOntology({v: v for v in prompts})

        detections_list = []

        for _, description in enumerate(ontology.prompts()):
            detections = self.grounding_dino_model.predict_with_classes(
                image=image,
                classes=[description],
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )

            detections_list.append(detections)

        detections = combine_detections(
            detections_list, overwrite_class_ids=range(len(detections_list))
        )

        return detections
