# -*- coding: utf-8 -*-
"""
Single source of truth for TB-CAD *private* DICOM tags in group (0011) and related helpers.

Design goals
------------
- Phase I / II / III must each have a **non-overlapping allocation plan** in the *same* odd
  private group, so that embedding Phase III does not wipe Phase I/II: we **never** call
  ``del``/``pop``/``clear`` on the dataset; we only **add or overwrite the exact
  (group, element) pairs** listed for Phase III. If P1/P2 use different elements, they remain.
- **Convention (recommended slots for future P1/P2 tools)** — *not* written by
  ``embed_tb_labels_into_dicom`` unless you add a separate tool:
  - (0011,0x0F20–0x0F2F)  Phase I summary / flags (US/LO) — *reserved by convention*
  - (0011,0x1000–0x10FF)  Phase II or legacy *reserved* (avoid using 1101–1106 for non-P3)
- **Phase III (multi-label) — canonical, written by embed_tb_labels_into_dicom**:
  - Private Creator at (0011,0x10) = TB_PHASE3_LABELS (a second LO at 0x11+ if 0x10 is taken)
  - D1..D6: (0011,0x1101)–(0011,0x1106)  VR=US  0/1 each — **six independent data elements**
  - Optional derived: (0011,0x1110) US, (0011,0x1111) CS
  - Optional **legacy** packed string: (0011,0x1010) LO — *only* with --write-packed-vector-lo
    (off by default so we do not replace arbitrary LO at 0x1010 on re-encode).

If your site already used (0011,0x1101)–(0x1106) for something else, either relocate Phase III
(see ``EMBED_GROUP`` / future CLI) or stop writing those elements from the other tool.
"""
from __future__ import annotations

from pydicom.tag import Tag

# --- One odd private group (default). All TB-CAD phases share this *group*; *elements* differ. ---
EMBED_GROUP = 0x0011

# Private creator for Phase III block (D1–D6, derived, etc.)
PHASE3_PRIVATE_CREATOR_ELEM = 0x0010
PHASE3_PRIVATE_CREATOR_VALUE = "TB_PHASE3_LABELS"

# Phase III: six independent US values (0/1)
TAG_D1 = Tag(EMBED_GROUP, 0x1101)
TAG_D2 = Tag(EMBED_GROUP, 0x1102)
TAG_D3 = Tag(EMBED_GROUP, 0x1103)
TAG_D4 = Tag(EMBED_GROUP, 0x1104)
TAG_D5 = Tag(EMBED_GROUP, 0x1105)  # cavitary
TAG_D6 = Tag(EMBED_GROUP, 0x1106)  # NTM

# Optional derived (Phase III)
TAG_SCORE = Tag(EMBED_GROUP, 0x1110)  # sum(D1..D4)
TAG_FINAL = Tag(EMBED_GROUP, 0x1111)  # CS TB|NTM

# Optional legacy: single LO backslash-packed vector; use only for old readers
TAG_PACKED_VEC_LO = Tag(EMBED_GROUP, 0x1010)

# Suggested (convention) — Phase I/II: **do not** use 0x1101–0x1106 for non-P3 data
PHASE1_ELEMENT_RANGE_HINT = (0x0F20, 0x0F2F)  # inclusive hex within EMBED_GROUP
PHASE2_ELEMENT_RANGE_HINT = (0x1000, 0x10FF)


def all_phase3_multilabel_tags() -> frozenset[Tag]:
    return frozenset({TAG_D1, TAG_D2, TAG_D3, TAG_D4, TAG_D5, TAG_D6})


def is_phase3_multilabel_tag(t: Tag) -> bool:
    return t in all_phase3_multilabel_tags()
