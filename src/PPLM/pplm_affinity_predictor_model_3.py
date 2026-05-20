import os
import torch
from .pplm_affinity.model import PPLM_Affinity


class PPLMPredictor:
    def __init__(self, gpu_id=0):
        """
        PPLM Affinity predictor (model3 only)
        """
        self.device = "cpu"  # 必要ならGPUに戻せる

        script_dir = os.path.dirname(os.path.abspath(__file__))

        # model_cv3.pkl のみ指定
        self.model_path = os.path.join(
            script_dir,
            "pplm_affinity/models/model_cv3.pkl"
        )

        self._load_model()

    def _load_model(self):
        """Load only model3"""
        checkpoint = torch.load(self.model_path, map_location=self.device)

        pplm_model_param = checkpoint["pplm_param"]
        model_state = checkpoint["model_state_dict"]

        self.model = PPLM_Affinity(pplm_model_param)
        self.model.load_state_dict(model_state)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, seqA: str, seqB: str) -> float:

        def clean_sequence(seq):
            seq = seq.replace(" ", "")
            seq = seq.replace("<start>", "")
            seq = seq.replace("<eos>", "")
            return seq.strip()

        seqB = clean_sequence(seqB)

        pred1 = self.model(seqA, seqB, self.device)
        pred2 = self.model(seqB, seqA, self.device)
        pred = (pred1 + pred2) / 2

        return pred.squeeze().cpu().item()