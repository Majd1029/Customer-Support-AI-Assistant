"""
cleaner.py — nettoyage des blocs de texte extraits.

Ce module prend la sortie brute du parser (text_blocks) et la nettoie :

  1. Déduplication   — supprime les blocs identiques ou quasi-identiques
                       (cas fréquent avec les PDFs double-colonne ou l'OCR)
  2. Bruit           — retire les blocs trop courts, purement numériques,
                       ou constitués uniquement de caractères spéciaux
  3. Fragmentation   — recolle les lignes qui semblent être la suite
                       d'une phrase coupée (commence en minuscule, pas de
                       ponctuation finale sur la ligne précédente)
  4. Normalisation   — espaces/tabulations multiples, tirets de césure,
                       caractères de contrôle
  5. En-têtes/pieds  — détecte et retire les headers/footers répétitifs

Usage autonome :
    from cleaner import clean_blocks
    clean = clean_blocks(raw_blocks)

Ou via CLI :
    python cleaner.py --input result.json --output result_clean.json
    python cleaner.py --input result.json --output result_clean.json --stats
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from loguru import logger


# ── Configuration ──────────────────────────────────────────────────────────────

# Longueur minimale pour qu'un bloc soit conservé (en caractères)
MIN_BLOCK_CHARS: int = 15

# Un bloc ne contenant que des chiffres/ponctuation est du bruit
_NOISE_RE = re.compile(r'^[\d\s\.\,\;\:\!\?\-\—\–\(\)\[\]\{\}\/\\\|\'\"«»""''…†§%°#@&*+=~^`]+$')

# Fin de ligne "ouverte" = phrase probablement coupée
_OPEN_END_RE  = re.compile(r'[a-zA-ZÀ-ÿ\d]$')    # finit par lettre ou chiffre
_HYPHEN_END_RE = re.compile(r'-\s*$')              # finit par tiret (césure)

# Début de fragment = commence par minuscule (hors chiffres, sigles courts)
_LOWER_START_RE = re.compile(r'^[a-zàâéèêëîïôùûüÿœæç]')

# Patterns pour les numéros de page isolés
_PAGE_NUM_RE = re.compile(r'^\s*\d{1,4}\s*$')

# Caractères de contrôle sauf \n \t
_CTRL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

# Tirets de césure en fin de ligne (mot-\ncoupé → motcoupé)
_HYPHEN_BREAK_RE = re.compile(r'(\w)-\s*\n\s*(\w)')

# Espaces multiples
_MULTI_SPACE_RE = re.compile(r'[ \t]{2,}')

# Ratio de similarité au-dessus duquel deux blocs sont considérés dupliqués
SIMILARITY_THRESHOLD: float = 0.92

# Nombre d'occurrences d'un bloc dans le document pour le considérer
# comme header/footer répétitif
HEADER_FOOTER_MIN_OCCURRENCES: int = 3


def _normalize_whitespace(text: str) -> str:
    """Supprime les caractères de contrôle et normalise les espaces."""
    text = _CTRL_RE.sub("", text)
    text = _HYPHEN_BREAK_RE.sub(r'\1\2', text)   # recoller les mots coupés par tiret
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def _is_noise(block: str) -> bool:
    """Retourne True si le bloc est du bruit pur."""
    stripped = block.strip()
    if not stripped:
        return True
    if len(stripped) < MIN_BLOCK_CHARS:
        return True
    if _PAGE_NUM_RE.match(stripped):
        return True
    if _NOISE_RE.match(stripped):
        return True
    return False


def _similarity(a: str, b: str) -> float:
    """Ratio de similarité entre deux chaînes (0–1)."""
    return SequenceMatcher(None, a[:500], b[:500]).ratio()


def _deduplicate(blocks: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """
    Supprime les blocs dupliqués ou quasi-dupliqués.
    Conserve le numéro de page du premier bloc vu.
    """
    seen_exact: set[str] = set()
    unique: list[tuple[int, str]] = []
    for pg, b in blocks:
        key = b.lower().strip()
        if key not in seen_exact:
            seen_exact.add(key)
            unique.append((pg, b))

    result: list[tuple[int, str]] = []
    window = 6
    for i, (pg, block) in enumerate(unique):
        is_dup = False
        start  = max(0, i - window)
        for _, prev in result[start:]:
            if _similarity(block, prev) >= SIMILARITY_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            result.append((pg, block))

    removed = len(blocks) - len(result)
    if removed:
        logger.debug(f"  déduplication : {removed} blocs supprimés")
    return result


def _detect_headers_footers(blocks: list[tuple[int, str]]) -> set[str]:
    """
    Identifie les blocs qui apparaissent suffisamment souvent pour être
    des headers ou footers.
    """
    counter: Counter = Counter()
    for _, b in blocks:
        key = re.sub(r'\s+', ' ', b.strip().lower())
        if len(key) < 80:
            counter[key] += 1
    return {k for k, cnt in counter.items() if cnt >= HEADER_FOOTER_MIN_OCCURRENCES}


def _rejoin_fragments(blocks: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """
    Recolle les fragments de phrases coupées entre deux blocs consécutifs.
    Lors de la fusion, conserve le numéro de page du bloc précédent.
    """
    if not blocks:
        return blocks

    result   = [blocks[0]]
    rejoined = 0

    for pg, block in blocks[1:]:
        prev_pg, prev = result[-1]
        prev_stripped = prev.rstrip()
        cur_stripped  = block.lstrip()

        if not cur_stripped:
            result.append((pg, block))
            continue

        prev_ends_open   = _OPEN_END_RE.search(prev_stripped) is not None
        prev_ends_hyphen = _HYPHEN_END_RE.search(prev_stripped) is not None
        cur_starts_lower = _LOWER_START_RE.match(cur_stripped) is not None
        same_page        = (prev_pg == pg)   # never rejoin across page boundaries

        if same_page and prev_ends_hyphen:
            joined = prev_stripped.rstrip('-').rstrip() + cur_stripped
            result[-1] = (prev_pg, joined)
            rejoined += 1
        elif same_page and prev_ends_open and cur_starts_lower:
            result[-1] = (prev_pg, prev_stripped + " " + cur_stripped)
            rejoined += 1
        else:
            result.append((pg, block))

    if rejoined:
        logger.debug(f"  rejoin : {rejoined} fragments recollés")
    return result


def _normalize_blocks(blocks: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Applique la normalisation des espaces sur chaque bloc."""
    normalized = []
    for pg, b in blocks:
        n = _normalize_whitespace(b)
        if n:
            normalized.append((pg, n))
    return normalized


def clean_blocks(
    blocks: list[tuple[int, str]],
    *,
    deduplicate: bool = True,
    remove_headers_footers: bool = True,
    rejoin_fragments: bool = True,
    min_chars: Optional[int] = None,
) -> list[tuple[int, str]]:
    """
    Pipeline complet de nettoyage des blocs de texte.

    Args:
        blocks                 : liste de (page, texte) sortie du parser
        deduplicate            : activer la déduplication
        remove_headers_footers : activer la détection header/footer
        rejoin_fragments       : activer le recollage de fragments
        min_chars              : override de MIN_BLOCK_CHARS (None = valeur par défaut)

    Returns:
        Liste de (page, texte) nettoyés
    """
    if not blocks:
        return blocks

    min_c = min_chars if min_chars is not None else MIN_BLOCK_CHARS
    stats_in = len(blocks)

    # 1. Normalisation espaces / caractères de contrôle
    blocks = _normalize_blocks(blocks)

    # 2. Retrait du bruit évident
    if min_chars is not None:
        blocks = [(pg, b) for pg, b in blocks if len(b.strip()) >= min_c and not _is_noise(b)]
    else:
        blocks = [(pg, b) for pg, b in blocks if not _is_noise(b)]

    # 3. Détection headers/footers avant déduplication
    hf_keys: set[str] = set()
    if remove_headers_footers:
        hf_keys = _detect_headers_footers(blocks)
        before  = len(blocks)
        blocks  = [
            (pg, b) for pg, b in blocks
            if re.sub(r'\s+', ' ', b.strip().lower()) not in hf_keys
        ]
        removed_hf = before - len(blocks)
        if removed_hf:
            logger.debug(f"  headers/footers : {removed_hf} blocs supprimés")

    # 4. Déduplication
    if deduplicate:
        blocks = _deduplicate(blocks)

    # 5. Recollage des fragments
    if rejoin_fragments:
        blocks = _rejoin_fragments(blocks)

    # 6. Nettoyage final après rejoin (espaces redondants)
    blocks = _normalize_blocks(blocks)
    blocks = [(pg, b) for pg, b in blocks if not _is_noise(b)]

    stats_out = len(blocks)
    logger.info(
        f"  clean_blocks : {stats_in} → {stats_out} blocs "
        f"({stats_in - stats_out} supprimés)"
    )

    return blocks


def clean_extraction_result(result) -> None:
    """
    Nettoie les text_blocks d'un ExtractionResult en place.

    Args:
        result : ExtractionResult (from models.py)
    """
    original = len(result.text_blocks)
    result.text_blocks = clean_blocks(result.text_blocks)
    logger.info(
        f"  '{result.source_file}': {original} → {len(result.text_blocks)} blocs "
        f"après nettoyage"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _process_json(
    input_path: Path,
    output_path: Path,
    show_stats: bool,
    min_chars: Optional[int],
) -> None:
    """
    Nettoie le champ text_blocks d'un fichier JSON produit par extract.py.

    Supporte deux formats :
    - objet unique  : { "source_file": ..., "text_blocks": [...], ... }
    - liste d'objets: [ { "source_file": ..., "text_blocks": [...] }, ... ]
    """
    data = json.loads(input_path.read_text(encoding="utf-8"))
    is_list = isinstance(data, list)
    docs    = data if is_list else [data]

    for doc in docs:
        if "text_blocks" not in doc:
            continue

        original = doc["text_blocks"]
        cleaned  = clean_blocks(
            original,
            min_chars=min_chars,
        )
        doc["text_blocks"] = cleaned

        if show_stats:
            removed = len(original) - len(cleaned)
            print(
                f"[{doc.get('source_file', '?')}] "
                f"{len(original)} → {len(cleaned)} blocs "
                f"({removed} supprimés, "
                f"{len(cleaned)/max(1,len(original))*100:.0f}% conservés)"
            )

    output_path.write_text(
        json.dumps(data if not is_list else docs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Résultat nettoyé → {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Nettoie les blocs de texte extraits (déduplication, bruit, fragments)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--input",     required=True, help="Fichier JSON produit par extract.py")
    ap.add_argument("--output",    required=True, help="Fichier JSON nettoyé à produire")
    ap.add_argument("--stats",     action="store_true", help="Afficher les statistiques de nettoyage")
    ap.add_argument("--min-chars", type=int, default=None,
                    help=f"Longueur minimale d'un bloc (défaut: {MIN_BLOCK_CHARS})")
    ap.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Désactiver la déduplication",
    )
    ap.add_argument(
        "--no-headers-footers",
        action="store_true",
        help="Désactiver la suppression des headers/footers répétitifs",
    )
    ap.add_argument(
        "--no-rejoin",
        action="store_true",
        help="Désactiver le recollage des fragments de phrases",
    )
    args = ap.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[ERREUR] Fichier introuvable : {input_path}", file=sys.stderr)
        sys.exit(1)

    _process_json(input_path, output_path, args.stats, args.min_chars)


if __name__ == "__main__":
    main()