"""
EggVision - Renomeação segura do raw_dataset
============================================
Renomeia imagens de uma classe para o padrão ovoXX-foto1.ext.

Uso:
  python src/rename_dataset.py --class normal --dry-run
  python src/rename_dataset.py --class normal --apply
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import PROJECT_CLASSES, VALID_IMAGE_EXTENSIONS, get_project_root


TEMP_MARKER = ".__eggvision_tmp_rename__"


def collect_images(class_path: str) -> list[str]:
    """Coleta imagens recursivamente em ordem determinística."""
    images = []

    for current_dir, _, filenames in os.walk(class_path):
        for filename in sorted(filenames):
            if TEMP_MARKER in filename:
                continue

            filepath = os.path.join(current_dir, filename)
            if not os.path.isfile(filepath):
                continue

            _, ext = os.path.splitext(filename)
            if ext.lower() not in VALID_IMAGE_EXTENSIONS:
                continue

            images.append(filepath)

    return sorted(images, key=lambda path: os.path.relpath(path, class_path).lower())


def build_plan(class_name: str) -> list[dict]:
    """Monta plano old -> temp -> new sem executar alterações."""
    root = get_project_root()
    class_path = os.path.join(root, "raw_dataset", class_name)

    if not os.path.isdir(class_path):
        print(f"\n❌ Pasta não encontrada: {class_path}")
        sys.exit(1)

    images = collect_images(class_path)
    width = max(2, len(str(len(images))))
    plan = []

    for index, old_path in enumerate(images, start=1):
        directory = os.path.dirname(old_path)
        _, ext = os.path.splitext(old_path)
        new_name = f"ovo{index:0{width}d}-foto1{ext.lower()}"
        new_path = os.path.join(directory, new_name)
        temp_name = f"{TEMP_MARKER}{index:0{width}d}{ext.lower()}"
        temp_path = os.path.join(directory, temp_name)

        plan.append({
            "index": index,
            "old": old_path,
            "temp": temp_path,
            "new": new_path,
        })

    return plan


def validate_plan(plan: list[dict]) -> None:
    """Garante que o plano não sobrescreve arquivos existentes."""
    old_paths = {item["old"] for item in plan}
    new_paths = [item["new"] for item in plan]
    temp_paths = [item["temp"] for item in plan]

    if len(new_paths) != len(set(new_paths)):
        print("\n❌ Plano inválido: nomes finais duplicados.")
        sys.exit(1)

    if len(temp_paths) != len(set(temp_paths)):
        print("\n❌ Plano inválido: nomes temporários duplicados.")
        sys.exit(1)

    conflicts = []
    for item in plan:
        if item["new"] != item["old"] and os.path.exists(item["new"]):
            if item["new"] not in old_paths:
                conflicts.append(item["new"])
        if os.path.exists(item["temp"]):
            conflicts.append(item["temp"])

    if conflicts:
        print("\n❌ A renomeação foi bloqueada para evitar sobrescrita:")
        for path in sorted(set(conflicts)):
            print(f"   - {path}")
        sys.exit(1)


def print_report(plan: list[dict], class_name: str, apply: bool) -> None:
    """Mostra relatório old -> new."""
    root = get_project_root()
    mode = "APLICAÇÃO REAL" if apply else "DRY-RUN"

    print("=" * 72)
    print(f"   EggVision - Renomeação {mode}")
    print("=" * 72)
    print(f"Classe: {class_name}")
    print(f"Imagens: {len(plan)}")
    print()

    if not plan:
        print("Nenhuma imagem encontrada.")
        return

    for item in plan:
        old_rel = os.path.relpath(item["old"], root)
        new_rel = os.path.relpath(item["new"], root)
        status = "sem alteração" if item["old"] == item["new"] else "renomear"
        print(f"{item['index']:03d}. {old_rel} -> {new_rel} ({status})")

    if not apply:
        print("\nNenhum arquivo foi alterado. Use --apply para executar.")


def apply_plan(plan: list[dict]) -> None:
    """Executa renomeação em duas etapas temporárias."""
    renamed_to_temp = []

    try:
        for item in plan:
            if item["old"] == item["new"]:
                continue
            os.rename(item["old"], item["temp"])
            renamed_to_temp.append(item)

        for item in renamed_to_temp:
            os.rename(item["temp"], item["new"])
    except OSError as error:
        print(f"\n❌ Erro durante renomeação: {error}")
        print("Tentando restaurar arquivos já movidos para nomes temporários...")
        for item in reversed(renamed_to_temp):
            if os.path.exists(item["temp"]) and not os.path.exists(item["old"]):
                os.rename(item["temp"], item["old"])
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="EggVision - Renomeação do raw_dataset")
    parser.add_argument("--class", dest="class_name", required=True, choices=PROJECT_CLASSES)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Mostra o plano sem alterar arquivos")
    mode.add_argument("--apply", action="store_true", help="Executa a renomeação")
    args = parser.parse_args()

    plan = build_plan(args.class_name)
    validate_plan(plan)
    print_report(plan, args.class_name, args.apply)

    if args.apply:
        apply_plan(plan)
        print("\n✅ Renomeação concluída.")


if __name__ == "__main__":
    main()
