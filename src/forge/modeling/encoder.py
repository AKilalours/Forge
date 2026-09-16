"""FORGE-Base: one encoder, two heads.

    text -> tokenizer -> transformer encoder -> contextual embeddings
                                                   |              |
                                           document head      token head
                                          (human vs ai)  (human/assisted/generated)

Deliberately not a mixture of experts. Reproducing a production MoE adds engineering risk
without touching the research question, which is about which data to generate, not about
architecture. If the data thesis is right it should show up on a plain encoder.

The two heads share the encoder on purpose. Token-level supervision is a strong
regulariser for the document decision: a model forced to say WHERE the AI text is cannot
satisfy the loss with a document-level shortcut like "this reads formal, call it AI".

torch is imported lazily so the rest of the package, including the evaluation lab and all
the pure functions, works on a machine without it.
"""

from __future__ import annotations

from dataclasses import dataclass

from forge.common.schemas import TokenLabel

N_TOKEN_CLASSES = len(TokenLabel)
N_DOC_CLASSES = 2


@dataclass
class ForgeConfig:
    backbone: str = "microsoft/deberta-v3-base"
    max_length: int = 512
    stride: int = 384
    dropout: float = 0.1
    token_loss_weight: float = 0.5   # ablate this; 0.0 turns off the token head entirely
    doc_loss_weight: float = 1.0


def build_model(config: ForgeConfig):  # pragma: no cover - needs torch
    """Construct the model. Imports torch lazily and fails with a useful message."""
    try:
        import torch  # noqa: F401  # the import IS the availability probe
        from torch import nn
        from transformers import AutoConfig, AutoModel
    except ImportError as e:
        raise RuntimeError(
            "Training needs the `train` extra. Run: pip install -e '.[train]'"
        ) from e

    class ForgeBase(nn.Module):
        def __init__(self, cfg: ForgeConfig) -> None:
            super().__init__()
            self.cfg = cfg
            hf_cfg = AutoConfig.from_pretrained(cfg.backbone)
            self.encoder = AutoModel.from_pretrained(cfg.backbone, config=hf_cfg)
            hidden = hf_cfg.hidden_size
            self.dropout = nn.Dropout(cfg.dropout)
            self.doc_head = nn.Linear(hidden, N_DOC_CLASSES)
            self.token_head = nn.Linear(hidden, N_TOKEN_CLASSES)

        def forward(self, input_ids, attention_mask, doc_labels=None, token_labels=None):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            h = self.dropout(out.last_hidden_state)          # (B, T, H)

            # Masked mean pooling, not [CLS]. DeBERTa has no NSP-pretrained [CLS], and
            # mean pooling makes the document score an average over positions, which is
            # consistent with how the token head sees the window.
            mask = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

            doc_logits = self.doc_head(pooled)               # (B, 2)
            token_logits = self.token_head(h)                # (B, T, 3)

            loss = None
            if doc_labels is not None:
                ce = nn.CrossEntropyLoss()
                loss = self.cfg.doc_loss_weight * ce(doc_logits, doc_labels)
                if token_labels is not None and self.cfg.token_loss_weight > 0:
                    tce = nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss + self.cfg.token_loss_weight * tce(
                        token_logits.reshape(-1, N_TOKEN_CLASSES), token_labels.reshape(-1)
                    )
            return {"loss": loss, "doc_logits": doc_logits, "token_logits": token_logits}

    return ForgeBase(config)
