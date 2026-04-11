from __future__ import annotations

import os
import numpy as np
import torch
from transformers import DistilBertModel, DistilBertTokenizerFast


LABEL_PROTOTYPES = {
    "news": "breaking news report journalism newsroom eyewitness verification current affairs",
    "meme": "meme joke humor parody satire viral post internet banter template",
    "propaganda / misinformation": "propaganda misinformation disinformation hoax manipulated claim false narrative misleading media",
}


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


class ContextClassifier:
    def __init__(self, device: str = "auto", model_name: str = "distilbert-base-uncased") -> None:
        self.requested_device = device
        self.model_name = model_name
        self._device: str | None = None
        self._tokenizer: DistilBertTokenizerFast | None = None
        self._model: DistilBertModel | None = None
        self._label_matrix: np.ndarray | None = None
        self._label_names = list(LABEL_PROTOTYPES.keys())
        self._load_error: Exception | None = None

    def _resolve_device(self) -> str:
        if self.requested_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.requested_device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return self.requested_device

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = self._resolve_device()
        return self._device

    def _load(self) -> None:
        if self._load_error is not None:
            return
        if self._tokenizer is None or self._model is None:
            try:
                self._tokenizer = self._load_tokenizer()
                self._model = self._load_model().to(self.device)
                self._model.eval()
            except Exception as exc:  # pragma: no cover - network/model safety
                self._load_error = exc
                return
        if self._label_matrix is None and self._tokenizer is not None and self._model is not None:
            label_embeddings = self._encode_texts_loaded(list(LABEL_PROTOTYPES.values()))
            self._label_matrix = np.vstack([_normalize(vector) for vector in label_embeddings])

    def _load_tokenizer(self) -> DistilBertTokenizerFast:
        try:
            return DistilBertTokenizerFast.from_pretrained(self.model_name, local_files_only=True)
        except OSError:
            if os.getenv("DEEPFAKE_ALLOW_MODEL_DOWNLOADS") == "1":
                return DistilBertTokenizerFast.from_pretrained(self.model_name)
            raise

    def _load_model(self) -> DistilBertModel:
        try:
            return DistilBertModel.from_pretrained(
                self.model_name,
                local_files_only=True,
                use_safetensors=False,
            )
        except OSError:
            if os.getenv("DEEPFAKE_ALLOW_MODEL_DOWNLOADS") == "1":
                return DistilBertModel.from_pretrained(self.model_name, use_safetensors=False)
            raise

    def _encode_texts_loaded(self, texts: list[str]) -> np.ndarray:
        assert self._tokenizer is not None and self._model is not None
        encoded = self._tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = self._model(**encoded)
        hidden = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * attention_mask).sum(dim=1) / torch.clamp(attention_mask.sum(dim=1), min=1)
        return pooled.detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def _heuristic_scores(text: str) -> dict[str, float]:
        lowered = text.lower()
        scores = {
            "news": 0.2,
            "meme": 0.2,
            "propaganda / misinformation": 0.2,
        }
        if any(token in lowered for token in ("breaking", "report", "article", "journalist", "news", "explainer")):
            scores["news"] += 0.5
        if any(token in lowered for token in ("meme", "joke", "satire", "parody", "funny", "viral")):
            scores["meme"] += 0.5
        if any(token in lowered for token in ("hoax", "false", "misleading", "propaganda", "fake", "disinformation")):
            scores["propaganda / misinformation"] += 0.5
        total = sum(scores.values())
        return {label: float(value / total) for label, value in scores.items()}

    def classify(self, text: str) -> tuple[str, dict[str, float]]:
        text = text.strip()
        if not text:
            default_scores = {"news": 0.34, "meme": 0.33, "propaganda / misinformation": 0.33}
            return "news", default_scores

        self._load()
        if self._load_error is not None or self._tokenizer is None or self._model is None or self._label_matrix is None:
            scores = self._heuristic_scores(text)
            return max(scores, key=scores.get), scores

        text_embedding = _normalize(self._encode_texts_loaded([text])[0])
        assert self._label_matrix is not None
        similarities = self._label_matrix @ text_embedding
        similarities = similarities - similarities.max()
        probabilities = np.exp(similarities)
        probabilities = probabilities / np.clip(probabilities.sum(), a_min=1e-6, a_max=None)
        scores = {label: float(prob) for label, prob in zip(self._label_names, probabilities, strict=False)}
        label = max(scores, key=scores.get)
        return label, scores
