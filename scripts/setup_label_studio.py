"""
Label Studio Project Setup Script
==================================
Automates creating the BSR classification project in Label Studio,
importing your 65 audio files, and exporting annotations for training.

Usage:
    # 1. Start Label Studio first:
    #    pip install label-studio && label-studio start
    #
    # 2. Get your API token from: http://localhost:8080/user/account
    #
    # 3. Run this script:
    #    python scripts/setup_label_studio.py --token YOUR_TOKEN --audio_dir data/raw

Run:  python scripts/setup_label_studio.py --help
"""

import os
import sys
import json
import argparse
import glob
from pathlib import Path

# Label Studio SDK
try:
    from label_studio_sdk import Client
    from label_studio_sdk.data_manager import Filters, Column, Type, Operator
except ImportError:
    print("Install label-studio-sdk:  pip install label-studio-sdk")
    sys.exit(1)


# ─────────────────────────────────────────────
# Labeling interface XML
# Sub-classes (BSR, IP, Sunroof) let you later train a multi-class model
# but collapse to binary OK/NOK for now
# ─────────────────────────────────────────────
LABEL_CONFIG = """
<View>
  <Style>
    .lsf-main-content { background: #1a1a2e; }
    h1 { color: #e0e0e0; font-size: 14px; }
  </Style>

  <Header value="BSR Track Audio — Label each event region"/>

  <Audio name="audio" value="$audio" zoom="true" volume="true" speed="true"/>

  <Labels name="label" toName="audio" allowEmpty="true">
    <Label value="OK"          background="#27ae60" hotkey="1"/>
    <Label value="NOK_BSR"     background="#e74c3c" hotkey="2"/>
    <Label value="NOK_IP"      background="#e67e22" hotkey="3"/>
    <Label value="NOK_Sunroof" background="#9b59b6" hotkey="4"/>
    <Label value="Unsure"      background="#95a5a6" hotkey="5"/>
  </Labels>

  <Choices name="overall_quality" toName="audio" showInLine="true">
    <Choice value="OK"  />
    <Choice value="NOK" />
  </Choices>

  <TextArea name="notes" toName="audio"
            placeholder="Optional: describe noise character (intermittent, speed-dependent, etc.)"
            rows="2"/>
</View>
"""

# Keyboard shortcuts reminder shown in console
SHORTCUTS_HELP = """
Label Studio Keyboard Shortcuts:
  1 → OK           (green)
  2 → NOK_BSR      (red)    — buzz, squeak, rattle
  3 → NOK_IP       (orange) — instrument panel creak
  4 → NOK_Sunroof  (purple) — sunroof wind/rattle
  5 → Unsure       (gray)   — keep but low training weight

Workflow:
  1. Press SPACE to play audio
  2. Click + drag on waveform to create a region
  3. Press hotkey to label it
  4. Set overall_quality at top (OK/NOK for whole file)
  5. Press → arrow to go to next task
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Setup Label Studio for BSR audio project")
    parser.add_argument("--token",     required=True, help="Label Studio API token")
    parser.add_argument("--host",      default="http://localhost:8080", help="Label Studio host URL")
    parser.add_argument("--audio_dir", default="data/raw", help="Directory containing audio files")
    parser.add_argument("--project_name", default="BSR_Audio_Classification")
    parser.add_argument("--export_dir", default="data/annotations", help="Where to save exported annotations")
    parser.add_argument("--audio_server", default=None,
                        help="Base URL of HTTP audio server e.g. http://localhost:8090 "
                             "(recommended on Windows — run: python -m http.server 8090 "
                             "inside your audio_dir folder)")
    parser.add_argument("--action",
                        choices=["setup", "export", "status"],
                        default="setup",
                        help="setup=create project & import files | export=download annotations | status=print stats")
    return parser.parse_args()


def connect(host: str, token: str) -> Client:
    ls = Client(url=host, api_key=token)
    ls.check_connection()
    print(f"Connected to Label Studio at {host}")
    return ls


def get_or_create_project(ls: Client, name: str) -> object:
    """Return existing project or create a new one."""
    for proj in ls.get_projects():
        if proj.get_params()["title"] == name:
            print(f"Found existing project: '{name}'  (id={proj.id})")
            return proj

    proj = ls.start_project(
        title=name,
        label_config=LABEL_CONFIG,
        description="BSR track audio OK/NOK classification for vehicle NVH testing",
    )
    print(f"Created new project: '{name}'  (id={proj.id})")
    return proj


def import_audio_files(proj, audio_dir: str, audio_server: str = None) -> int:
    """Import audio files from directory into Label Studio project."""
    audio_dir = Path(audio_dir)
    extensions = ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]
    files = []
    for ext in extensions:
        files.extend(audio_dir.glob(ext))

    if not files:
        print(f"No audio files found in {audio_dir}")
        print("Expected extensions: wav, mp3, flac, ogg, m4a")
        return 0

    print(f"Found {len(files)} audio files — importing...")

    tasks = []
    for f in sorted(files):
        if audio_server:
            # HTTP server mode — works reliably on Windows, no local-files issues
            audio_url = f"{audio_server.rstrip('/')}/{f.name}"
        else:
            # Local files mode — may have issues on Windows
            forward_path = str(f.resolve()).replace("\\", "/")
            audio_url = f"/data/local-files/?d={forward_path}"

        tasks.append({
            "data": {
                "audio":     audio_url,
                "file_name": f.name,
                "file_id":   f.stem,
            }
        })

    proj.import_tasks(tasks)
    print(f"Imported {len(tasks)} tasks into Label Studio")
    print(SHORTCUTS_HELP)
    return len(tasks)


def export_annotations(proj, export_dir: str) -> Path:
    """Export completed annotations to JSON + CSV formats."""
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Full JSON export (Label Studio format)
    tasks = proj.export_tasks(export_type="JSON")
    json_path = export_dir / "annotations_labelstudio.json"
    with open(json_path, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"Saved {len(tasks)} tasks → {json_path}")

    # Convert to flat CSV for training pipeline
    csv_path = export_dir / "annotations.csv"
    rows = convert_to_training_csv(tasks, csv_path)
    print(f"Saved {rows} annotation rows → {csv_path}")

    return csv_path


def convert_to_training_csv(tasks: list, output_path: Path) -> int:
    """
    Convert Label Studio JSON export to flat CSV with columns:
      file_id, file_name, start_sec, end_sec, label, confidence, overall_quality, notes
    """
    import csv

    rows = []
    for task in tasks:
        file_id   = task["data"].get("file_id", "unknown")
        file_name = task["data"].get("file_name", "unknown")

        overall_quality = "UNKNOWN"
        notes = ""

        # Pull from annotations
        for annotation in task.get("annotations", []):
            for result in annotation.get("result", []):
                r_type = result.get("type", "")
                r_from = result.get("from_name", "")

                # Overall file quality (Choices widget)
                if r_type == "choices" and r_from == "overall_quality":
                    overall_quality = result["value"]["choices"][0]

                # Notes
                if r_type == "textarea" and r_from == "notes":
                    notes = result["value"].get("text", [""])[0]

                # Temporal regions (Labels widget)
                if r_type == "labels" and r_from == "label":
                    duration_sec = task.get("data", {}).get("duration", None)
                    start_frac   = result["value"]["start"]   # 0-1 fraction of file
                    end_frac     = result["value"]["end"]
                    label_value  = result["value"]["labels"][0] if result["value"]["labels"] else "Unsure"

                    # Convert fractions to absolute seconds if duration known
                    # Label Studio stores start/end as percentage (0-100) for audio
                    # Adjust based on actual export format
                    rows.append({
                        "file_id":         file_id,
                        "file_name":       file_name,
                        "start_frac":      start_frac,
                        "end_frac":        end_frac,
                        "label":           label_value,
                        "overall_quality": overall_quality,
                        "notes":           notes,
                    })

        # If no temporal regions but overall quality set → whole-file label
        if not any(r.get("file_id") == file_id for r in rows):
            rows.append({
                "file_id":         file_id,
                "file_name":       file_name,
                "start_frac":      0.0,
                "end_frac":        1.0,
                "label":           overall_quality if overall_quality != "UNKNOWN" else "Unsure",
                "overall_quality": overall_quality,
                "notes":           notes,
            })

    if not rows:
        print("Warning: no annotated tasks found. Complete annotations in Label Studio first.")
        return 0

    fieldnames = ["file_id", "file_name", "start_frac", "end_frac",
                  "label", "overall_quality", "notes"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def print_status(proj) -> None:
    """Print annotation progress stats."""
    params = proj.get_params()
    stats  = proj.get_stats()
    print(f"\nProject: {params['title']}  (id={proj.id})")
    print(f"  Total tasks:      {stats.get('total_tasks', '?')}")
    print(f"  Annotated tasks:  {stats.get('total_annotations', '?')}")
    print(f"  Skipped:          {stats.get('skipped_annotations', '?')}")
    completion = 0
    if stats.get("total_tasks", 0) > 0:
        completion = 100 * stats.get("total_annotations", 0) / stats["total_tasks"]
    print(f"  Completion:       {completion:.1f}%")


def main():
    args = parse_args()
    ls   = connect(args.host, args.token)
    proj = get_or_create_project(ls, args.project_name)

    if args.action == "setup":
        n = import_audio_files(proj, args.audio_dir, args.audio_server)
        print(f"\nDone. Open {args.host} → project '{args.project_name}' to start labeling.")
        print("When done, run:  python scripts/setup_label_studio.py --action export --token YOUR_TOKEN")

    elif args.action == "export":
        Path(args.export_dir).mkdir(parents=True, exist_ok=True)
        csv_path = export_annotations(proj, args.export_dir)
        print(f"\nAnnotations exported. Run training with:")
        print(f"  python scripts/train.py --annotations {csv_path}")

    elif args.action == "status":
        print_status(proj)


if __name__ == "__main__":
    main()
