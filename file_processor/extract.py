import argparse
import base64
import json
import sys
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from file_processor.parser    import extract, SUPPORTED
from file_processor.captioner import caption_all
from file_processor.cleaner   import clean_blocks
from file_processor.models    import ExtractionResult, ExtractedImage, ExtractedTable
from file_preparation.chunking import chunk_document

console = Console()


# ── Chunking — delegated to file_preparation/chunking/chunker.py ─────────────
# All chunking logic (token counting, sentence splitting, sliding window,
# text cleaning, metadata cleaning, slugification) lives there.

def _to_dict(result: ExtractionResult, include_images_b64: bool = False, doc_uid: str = "") -> dict:
    """Thin wrapper — delegates all chunking logic to the chunker module."""
    return chunk_document(result, include_images_b64=include_images_b64, doc_uid=doc_uid)


def _display(result: ExtractionResult) -> None:
    s = result.stats
    console.print(Panel(
        f"[bold]{result.source_file}[/bold]\n"
        f"  blocs texte : [cyan]{s['text_blocks']}[/cyan]\n"
        f"  tableaux    : [cyan]{s['tables']}[/cyan]\n"
        f"  images      : [cyan]{s['images']}[/cyan]"
        + (f"  (dont [green]{s['images_captioned']} captionnées[/green])"
           if s['images_captioned'] else ""),
        title="Résultat extraction",
        border_style="blue",
    ))

    if result.text_blocks:
        console.print("\n[bold yellow]── Blocs texte (aperçu 3 premiers)[/bold yellow]")
        for i, (pg, b) in enumerate(result.text_blocks[:3], 1):
            preview = b[:200] + "…" if len(b) > 200 else b
            console.print(f"  [{i}] (p.{pg+1}) {preview}")

    if result.tables:
        console.print("\n[bold yellow]── Tableaux[/bold yellow]")
        for t in result.tables:
            console.print(f"  Page {t.page_number+1}, tableau {t.table_index+1} "
                          f"— {len(t.raw_rows)} lignes")
            rows = t.raw_rows[:4]
            if rows:
                tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
                for col in rows[0]:
                    tbl.add_column(col, max_width=20)
                for row in rows[1:]:
                    tbl.add_row(*row)
                console.print(tbl)

    if result.images:
        console.print("\n[bold yellow]── Images[/bold yellow]")
        for img in result.images:
            size_kb = len(img.base64_data) * 3 // 4 // 1024
            cap = f" → {img.caption[:80]}…" if img.caption else " → [dim]pas de caption[/dim]"
            console.print(f"  Page {img.page_number+1}, image {img.image_index+1} "
                          f"[{size_kb} KB]{cap}")


def _save_images(images: list[ExtractedImage], out_dir: Path, source_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(source_name).stem
    for img in images:
        ext  = img.mime_type.split("/")[-1]
        name = f"{stem}_p{img.page_number+1}_i{img.image_index+1}.{ext}"
        dest = out_dir / name
        raw  = base64.b64decode(img.base64_data)
        dest.write_bytes(raw)
        logger.info(f"  image sauvegardée → {dest}")
    console.print(f"[green]{len(images)} image(s) sauvegardée(s) dans {out_dir}/[/green]")


def process_file(
    path: Path,
    *,
    caption: bool = True,
    caption_backend: str = "groq",    # "groq" (cloud, active) | "llava" (Ollama local)
    pdf_pass: str | None = None,
    clean: bool = True,
) -> ExtractionResult:
    """
    Extrait le contenu d'un fichier et retourne un ExtractionResult.

    Args:
        path             : chemin vers le fichier
        caption          : activer le captioning d'images (défaut: True)
                           Toujours activer avant l'indexation — les chunks image
                           sans caption embedent comme des vecteurs quasi-nuls.
        caption_backend  : "groq" (Llama-4-Scout via API, actif) ou
                           "llava" (Ollama local, optionnel)
        pdf_pass         : mot de passe pour les PDF chiffrés
        clean            : activer le nettoyage des blocs texte (défaut: True)
    """
    text_blocks, tables, images, doc_metadata = extract(path, pdf_pass=pdf_pass)

    # ── Nettoyage des blocs texte ─────────────────────────────────────────────
    if clean and text_blocks:
        before = len(text_blocks)
        # Formats where every block carries page=0 (no real page numbers):
        #   • DOCX — python-docx doesn't expose page boundaries; all paragraphs
        #            share page=0, so the same-page guard is always True and
        #            the rejoin heuristic would stitch across section boundaries.
        #   • EML  — email body blocks have no page structure at all; page=0
        #            throughout, so aggressive rejoining would incorrectly merge
        #            distinct paragraphs that happen to start with a lowercase word.
        # For both formats we keep deduplication and noise removal but skip
        # fragment rejoining to avoid cross-boundary merges.
        _ext = path.suffix.lower()
        text_blocks = clean_blocks(
            text_blocks,
            rejoin_fragments=_ext not in {".docx", ".eml"},
        )
        logger.info(
            f"  nettoyage : {before} → {len(text_blocks)} blocs "
            f"({before - len(text_blocks)} supprimés)"
        )

    if caption and images:
        images = caption_all(images, backend=caption_backend)

    return ExtractionResult(
        source_file=path.name,
        text_blocks=text_blocks,
        tables=tables,
        images=images,
        doc_metadata=doc_metadata,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extraction de texte, tableaux et images depuis un document",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("input",             help="Fichier ou dossier à traiter")
    ap.add_argument("--caption",         action="store_true",
                    help="Générer des captions pour les images (nécessite Ollama + llava)")
    ap.add_argument("--no-clean",        action="store_true",
                    help="Désactiver le nettoyage automatique des blocs texte")
    ap.add_argument("--output",          help="Sauvegarder le résultat en JSON (ex: result.json)")
    ap.add_argument("--save-images",     metavar="DIR",
                    help="Sauvegarder les images extraites dans ce dossier")
    ap.add_argument("--password",        help="Mot de passe pour les fichiers chiffrés")
    ap.add_argument("--quiet",           action="store_true", help="Pas d'affichage terminal")
    args = ap.parse_args()

    input_path = Path(args.input)
    do_clean   = not args.no_clean

    if input_path.is_dir():
        files = [
            f for f in input_path.rglob("*")
            if f.is_file()
            and f.suffix.lower() in SUPPORTED
            and not f.name.startswith("~")
        ]
        if not files:
            console.print(f"[red]Aucun fichier supporté trouvé dans {input_path}[/red]")
            sys.exit(1)

        console.print(f"[bold]{len(files)} fichier(s) trouvé(s)[/bold]")
        all_results = []
        for f in files:
            try:
                r = process_file(f, caption=args.caption,
                                 pdf_pass=args.password, clean=do_clean)
                if not args.quiet:
                    _display(r)
                if args.save_images:
                    _save_images(r.images, Path(args.save_images), r.source_file)
                all_results.append(_to_dict(r))
            except Exception as e:
                logger.error(f"Erreur sur {f.name}: {e}")

        if args.output:
            Path(args.output).write_text(
                json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            console.print(f"[green]Résultats sauvegardés → {args.output}[/green]")
        return

    if not input_path.exists():
        console.print(f"[red]Fichier introuvable: {input_path}[/red]")
        sys.exit(1)

    if input_path.suffix.lower() not in SUPPORTED:
        console.print(f"[red]Extension non supportée: {input_path.suffix}[/red]")
        console.print(f"Formats acceptés: {', '.join(sorted(SUPPORTED))}")
        sys.exit(1)

    result = process_file(input_path, caption=args.caption,
                          pdf_pass=args.password, clean=do_clean)

    if not args.quiet:
        _display(result)

    if args.save_images and result.images:
        _save_images(result.images, Path(args.save_images), result.source_file)

    if args.output:
        data = json.dumps(_to_dict(result), ensure_ascii=False, indent=2)
        Path(args.output).write_text(data, encoding="utf-8")
        console.print(f"[green]Résultat sauvegardé → {args.output}[/green]")


if __name__ == "__main__":
    main()