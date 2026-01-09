from __future__ import annotations

import html
import json
import re
import threading
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


# =============================================================================
# Text cleaning / filtering
# =============================================================================

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)

STOP_EXACT: set[str] = {
    "зависит от",
    "зависит",
    "расширяет",
    "extends",
    "depends on",
    "include",
    "extend",
}
STOP_PREFIXES: tuple[str, ...] = ("сервис ", "service ")
STOP_REGEX: tuple[str, ...] = (
    r"^\s*зависит\s+от\s*$",
    r"^\s*расширяет\s*$",
    r"^\s*depends\s+on\s*$",
    r"^\s*extends\s*$",
    r"^\s*include\s*$",
    r"^\s*extend\s*$",
)

# A small stoplist for extremely short connector-like labels.
_TINY_TOKENS_RE = re.compile(r"^(и|или|от|до|в|на|о|об|к|по|за|с|без)$", re.IGNORECASE)


def clean_text(text: str) -> str:
    """
    Remove HTML tags/artefacts and normalize whitespace.

    Notes:
    - draw.io/mxGraph exports may contain rich HTML labels.
    - Some labels may be split into multiple <font> tags, producing artefacts
      like "о т" instead of "от".
    """
    if not text:
        return ""

    s = html.unescape(text)

    # Preserve word boundaries for typical HTML label constructs
    s = re.sub(r"(?i)<\s*br\s*/?\s*>", " ", s)
    s = re.sub(
        r"(?i)</?\s*(div|p|tr|td|li|ul|ol|table|tbody|thead)\b[^>]*>",
        " ",
        s,
    )

    # Remove remaining tags
    s = _TAG_RE.sub(" ", s)

    # Normalize NBSP
    s = s.replace("\xa0", " ")

    # Fix common artefact: "<font>Зависит о</font><font>т</font>" -> "Зависит о т"
    s = re.sub(r"\bо\s+т\b", "от", s, flags=re.IGNORECASE)

    # Final whitespace normalization
    s = _WS_RE.sub(" ", s).strip()
    return s


def norm_key(text: str) -> str:
    """Normalize string for matching: clean HTML, lower, remove punctuation, normalize spaces."""
    s = clean_text(text).lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def is_valid_usecase(name: str) -> bool:
    """Heuristics to filter out non-use-case labels (service headers, relationship labels, etc.)."""
    n = norm_key(name)
    if not n or len(n) < 4:
        return False
    if n in STOP_EXACT:
        return False
    if any(n.startswith(p) for p in STOP_PREFIXES):
        return False
    if any(re.match(rx, n, flags=re.IGNORECASE) for rx in STOP_REGEX):
        return False
    if _TINY_TOKENS_RE.fullmatch(n):
        return False
    return True


# =============================================================================
# Embeddings
# =============================================================================

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Cache models by name (Streamlit may re-run; also avoids re-download/reload).
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}
_MODEL_LOCK = threading.Lock()


def get_model(model_name: str) -> SentenceTransformer:
    """Load (or reuse) a SentenceTransformer model by name."""
    with _MODEL_LOCK:
        model = _MODEL_CACHE.get(model_name)
        if model is None:
            model = SentenceTransformer(model_name)
            _MODEL_CACHE[model_name] = model
        return model


def embed_texts(
    texts: Sequence[str],
    model: SentenceTransformer,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Encode texts into L2-normalized embeddings (so cosine similarity = dot product).

    Returns:
        np.ndarray shape (N, D), dtype float32
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    embs = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(embs, dtype=np.float32)


def best_match(
    query_emb: np.ndarray,
    cand_embs: np.ndarray,
    candidates: Sequence[str],
) -> Tuple[str, float]:
    """
    Find best candidate by cosine similarity.

    Assumptions:
        - query_emb is normalized (L2)
        - cand_embs rows are normalized (L2)
        - cosine similarity = dot product
    """
    if cand_embs.size == 0 or not candidates:
        return "", 0.0

    sims = cand_embs @ query_emb  # (N,)
    idx = int(np.argmax(sims))
    return candidates[idx], float(sims[idx])


def adaptive_threshold(requirement: str, base: float) -> float:
    """
    Adaptive threshold by requirement length (domain-agnostic).

    Short requirements are checked stricter to reduce false positives.
    Long requirements may allow slightly lower threshold.
    """
    n_words = len(norm_key(requirement).split())
    if n_words <= 2:
        return min(0.75, base + 0.10)
    if n_words >= 10:
        return max(0.35, base - 0.05)
    return base


# =============================================================================
# Data extraction
# =============================================================================

def extract_requirements_from_text(text: str) -> List[str]:
    """Extract requirements from text lines starting with '-'."""
    reqs: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("-"):
            req = line[1:].strip()
            if req:
                reqs.append(req)
    return reqs


@dataclass(frozen=True)
class UmlData:
    actors: Dict[str, str]
    use_cases: Dict[str, str]
    relationships: List[Dict[str, str]]
    filtered_out: List[Tuple[str, str]]


def parse_uml_elements_from_xml(xml_bytes: bytes) -> UmlData:
    """Parse mxGraph/draw.io-like XML and extract actors, use-cases and relationships."""
    root = ET.fromstring(xml_bytes)

    actors: Dict[str, str] = {}
    use_cases: Dict[str, str] = {}
    relationships: List[Dict[str, str]] = []
    filtered_out: List[Tuple[str, str]] = []

    for cell in root.findall(".//mxCell"):
        cell_id = cell.get("id")
        if not cell_id:
            continue

        src = cell.get("source")
        tgt = cell.get("target")
        if src and tgt:
            relationships.append({"source": src, "target": tgt})

        raw_value = cell.get("value") or ""
        if not raw_value:
            continue

        style = cell.get("style", "")
        value = clean_text(raw_value)

        if "umlActor" in style or "shape=umlActor" in style:
            actors[cell_id] = value
        else:
            if is_valid_usecase(value):
                use_cases[cell_id] = value
            else:
                filtered_out.append((cell_id, value))

    return UmlData(
        actors=actors,
        use_cases=use_cases,
        relationships=relationships,
        filtered_out=filtered_out,
    )


def map_actors_to_use_cases(uml: UmlData) -> Dict[str, List[str]]:
    """
    Map actors to use-cases.

    Supports both directions:
        actor -> usecase and usecase -> actor
    """
    actor_use_cases: Dict[str, List[str]] = {}

    def add(actor: str, usecase: str) -> None:
        actor_use_cases.setdefault(actor, [])
        if usecase not in actor_use_cases[actor]:
            actor_use_cases[actor].append(usecase)

    for rel in uml.relationships:
        s = rel["source"]
        t = rel["target"]
        if s in uml.actors and t in uml.use_cases:
            add(uml.actors[s], uml.use_cases[t])
        elif t in uml.actors and s in uml.use_cases:
            add(uml.actors[t], uml.use_cases[s])

    # Stable order (useful for reproducible reports)
    for actor in list(actor_use_cases.keys()):
        actor_use_cases[actor] = sorted(actor_use_cases[actor])

    return dict(sorted(actor_use_cases.items(), key=lambda kv: kv[0]))


# =============================================================================
# Report + analysis
# =============================================================================

@dataclass(frozen=True)
class MatchInfo:
    requirement: str
    best_score: float
    best_candidate: str


@dataclass(frozen=True)
class Report:
    total_requirements: int
    covered_requirements: int
    missing_requirements: List[str]
    actors: Dict[str, List[str]]
    unmapped_use_cases: List[str]
    threshold_base: float
    model_name: str
    matches_for_missing: List[MatchInfo]
    filtered_out: List[Tuple[str, str]]


def analyze(
    requirements_text: str,
    xml_bytes: bytes,
    threshold: float = 0.55,
    model_name: str = DEFAULT_MODEL_NAME,
    include_actors_in_coverage: bool = True,
) -> Report:
    requirements = extract_requirements_from_text(requirements_text)
    uml = parse_uml_elements_from_xml(xml_bytes)
    actor_mapping = map_actors_to_use_cases(uml)

    candidates_set = set(uml.use_cases.values())
    if include_actors_in_coverage:
        candidates_set |= set(uml.actors.values())
    candidates = sorted(candidates_set)

    # Edge cases: empty inputs
    if not requirements:
        return Report(
            total_requirements=0,
            covered_requirements=0,
            missing_requirements=[],
            actors=actor_mapping,
            unmapped_use_cases=sorted(list(uml.use_cases.values())),
            threshold_base=threshold,
            model_name=model_name,
            matches_for_missing=[],
            filtered_out=uml.filtered_out,
        )
    if not candidates:
        return Report(
            total_requirements=len(requirements),
            covered_requirements=0,
            missing_requirements=list(requirements),
            actors=actor_mapping,
            unmapped_use_cases=[],
            threshold_base=threshold,
            model_name=model_name,
            matches_for_missing=[MatchInfo(r, 0.0, "") for r in requirements],
            filtered_out=uml.filtered_out,
        )

    model = get_model(model_name)

    req_norm = [norm_key(r) for r in requirements]
    cand_norm = [norm_key(c) for c in candidates]

    req_embs = embed_texts(req_norm, model)
    cand_embs = embed_texts(cand_norm, model)

    missing: List[str] = []
    missing_matches: List[MatchInfo] = []

    for i, req in enumerate(requirements):
        thr = adaptive_threshold(req, threshold)
        best_cand, best_score = best_match(req_embs[i], cand_embs, candidates)
        if best_score < thr:
            missing.append(req)
            missing_matches.append(MatchInfo(req, best_score, best_cand))

    mapped_usecases = {uc for ucs in actor_mapping.values() for uc in ucs}
    unmapped = sorted(list(set(uml.use_cases.values()) - mapped_usecases))

    return Report(
        total_requirements=len(requirements),
        covered_requirements=len(requirements) - len(missing),
        missing_requirements=missing,
        actors=actor_mapping,
        unmapped_use_cases=unmapped,
        threshold_base=threshold,
        model_name=model_name,
        matches_for_missing=missing_matches,
        filtered_out=uml.filtered_out,
    )


def format_report_text(report: Report, include_debug: bool = False) -> str:
    """Render report to human-readable plain text."""
    lines: List[str] = []
    lines.append(f"Общее количество требований: {report.total_requirements}")
    lines.append(f"Покрыто требований: {report.covered_requirements}")
    lines.append("")

    lines.append("Не покрытые требования:")
    if report.missing_requirements:
        for req in report.missing_requirements:
            lines.append(f"- {req}")
    else:
        lines.append("- (нет)")

    if include_debug and report.matches_for_missing:
        lines.append("")
        lines.append("Лучшие совпадения для непокрытых (debug):")
        for m in report.matches_for_missing:
            cand = m.best_candidate or "(нет)"
            lines.append(f"- {m.requirement} => {cand} (score={m.best_score:.3f})")

    lines.append("")
    lines.append("Соответствие акторов и прецедентов:")
    if report.actors:
        for actor, ucs in report.actors.items():
            lines.append("")
            lines.append(f"Актор: {actor}")
            for uc in ucs:
                lines.append(f"  - {uc}")
    else:
        lines.append("- (нет связей актор-прецедент)")

    if report.unmapped_use_cases:
        lines.append("")
        lines.append("Прецеденты без связей с акторами:")
        for uc in report.unmapped_use_cases:
            lines.append(f"- {uc}")

    if include_debug and report.filtered_out:
        lines.append("")
        lines.append("Отфильтрованные элементы (не считаются прецедентами):")
        for _id, val in report.filtered_out[:200]:
            lines.append(f"- {_id}: {val}")
        if len(report.filtered_out) > 200:
            lines.append(f"... (показаны первые 200 из {len(report.filtered_out)})")

    return "\n".join(lines)


def report_to_json(report: Report) -> str:
    """Serialize report to JSON (UTF-8, human-readable)."""
    return json.dumps(asdict(report), ensure_ascii=False, indent=2)
