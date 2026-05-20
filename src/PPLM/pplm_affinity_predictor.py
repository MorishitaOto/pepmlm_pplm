import os
import torch
from .pplm_affinity.model import PPLM_Affinity


class PPLMPredictor:
    def __init__(self, gpu_id=0):
        """
        PPLM Affinity predictor (models are loaded once)

        Parameters
        ----------
        gpu_id : int
            GPU id to use
        """
        assigned_device = f"cuda:{gpu_id}"
        self.device = assigned_device if torch.cuda.is_available() else "cpu"
        #self.device = "cpu"

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_path = [
            os.path.join(
                script_dir,
                f"pplm_affinity/models/model_cv{i}.pkl"
            )
            for i in range(5)
        ]

        self.models = []
        self._load_models()

    def _load_models(self):
        """Load all PPLM models once onto GPU"""
        for model_path in self.models_path:
            checkpoint = torch.load(model_path, map_location=self.device)

            pplm_model_param = checkpoint["pplm_param"]
            model_state = checkpoint["model_state_dict"]

            model = PPLM_Affinity(pplm_model_param)
            model.load_state_dict(model_state)
            model.to(self.device)
            model.eval()  # ← 超重要

            self.models.append(model)

    @torch.no_grad()
    def predict(self, seqA: str, seqB: str) -> float:
        
        def clean_sequence(seq):
            seq = seq.replace(" ", "")
            seq = seq.replace("<start>", "")
            seq = seq.replace("<eos>", "")
            return seq.strip()
        seqB = clean_sequence(seqB)
        
        """
        Predict binding affinity for two sequences

        Parameters
        ----------
        seqA : str
            receptor sequence
        seqB : str
            ligand sequence

        Returns
        -------
        float
            predicted binding affinity
        """
        predictions_list = []

#all models
        for model in self.models:
            pred1 = model(seqA, seqB, self.device)
            pred2 = model(seqB, seqA, self.device)
            pred = (pred1 + pred2) / 2
            predictions_list.append(pred)

        predictions = torch.stack(predictions_list)
        predictions = torch.mean(predictions, dim=0)

        return predictions.squeeze().cpu().item()
