# Quick Python re-packager + inspector:
# 1. Pack the Healthcare AI project into tar.gz via Python (UTF-8 safe).
# 2. Inspect the archive's headers - print first 5 and 5 random entries
#    with their raw UTF-8 name.
#
# Usage:
#   python "e:\各种PY程序\18-医疗AI模型系统\hpc\pack_tar_python.py"

import os
import sys
import tarfile
import random
import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXCLUDE_DIRS = {"outputs", "outputs_hpc", "logs_hpc", "env_py311", "__pycache__"}
EXCLUDE_EXTS = {".pyc", ".log"}
EXCLUDE_NAMES = {"__pycache__"}


def should_exclude(path: str) -> bool:
    rel = os.path.relpath(path, PROJECT_ROOT).replace("\\", "/")
    parts = rel.split("/")
    if any(p in EXCLUDE_DIRS for p in parts):
        return True
    name = parts[-1]
    if name in EXCLUDE_NAMES:
        return True
    ext = os.path.splitext(name)[1].lower()
    if ext in EXCLUDE_EXTS:
        return True
    return False


def main() -> int:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tar_path = os.path.join(os.environ.get("TEMP", "."), f"healthcare_hpc_py_{stamp}.tar.gz")

    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Target archive: {tar_path}")

    project_folder = os.path.basename(PROJECT_ROOT)
    parent = os.path.dirname(PROJECT_ROOT)

    count_files = 0
    count_dirs = 0

    with tarfile.open(tar_path, "w:gz", encoding="utf-8") as tar:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # filter excluded directories in-place so os.walk skips them.
            dirs[:] = [d for d in dirs if os.path.join(root, d) and not should_exclude(os.path.join(root, d))]
            for fname in files:
                fpath = os.path.join(root, fname)
                if should_exclude(fpath):
                    continue
                abs_root = os.path.join(parent, project_folder)
                # Archive path under "18-医疗AI模型系统/..." prefix (UTF-8)
                rel_from_folder = os.path.relpath(fpath, PROJECT_ROOT).replace("\\", "/")
                arcname = f"{project_folder}/{rel_from_folder}"
                tar.add(fpath, arcname=arcname, recursive=False)
                count_files += 1
            # add the directories themselves so `tar -xzf` creates them.
            for dname in dirs:
                dpath = os.path.join(root, dname)
                rel_from_folder = os.path.relpath(dpath, PROJECT_ROOT).replace("\\", "/")
                arcname = f"{project_folder}/{rel_from_folder}/"
                tar.add(dpath, arcname=arcname, recursive=False)
                count_dirs += 1

    size_mb = round(os.path.getsize(tar_path) / 1024 / 1024, 3)
    print(f"[OK ] Packed {count_files} files + {count_dirs} dirs -> {tar_path} ({size_mb} MB)")

    # --- Inspect headers with Python (bypasses CP936 console limits) ---
    print("\n[--- Archive header inspection (raw UTF-8) ---]")
    with tarfile.open(tar_path, "r:gz", encoding="utf-8") as tar:
        members = tar.getmembers()
        total = len(members)
        print(f"Total members: {total}")
        sample = members[:5] + random.sample(members, min(5, total - 5)) if total > 10 else members
        for m in sample:
            kind = "DIR" if m.isdir() else ("FILE" if m.isfile() else "LINK")
            print(f"  [{kind:4s}] {m.name}  (size={m.size})")

        # Confirm folder name itself has Chinese characters preserved.
        project_folder_members = [m for m in members if m.name.split("/", 1)[0] == project_folder]
        print(f"\nFolder '{project_folder}' preserved in archive -> {len(project_folder_members)} entries found.")

    # cleanup - keep for external scp, remove when done
    print(f"\n[OK ] Archive preserved. You can scp it to the HPC node, or delete it after upload.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
