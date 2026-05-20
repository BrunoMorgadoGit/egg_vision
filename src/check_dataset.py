"""
EggVision - Verificação do Dataset
====================================
Verifica a estrutura do dataset e exibe estatísticas detalhadas.

Uso:
  python src/check_dataset.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import get_project_root, count_images_by_class, VALID_IMAGE_EXTENSIONS


def check_dataset():
    """Verifica e exibe estatísticas do raw_dataset e dataset."""
    print("=" * 60)
    print("   🥚 EggVision — Verificação do Dataset")
    print("=" * 60)

    root = get_project_root()
    raw_path = os.path.join(root, "raw_dataset")
    dataset_path = os.path.join(root, "dataset")
    warnings = []

    # ---- RAW DATASET ----
    print(f"\n📁 Raw Dataset: {raw_path}")
    if os.path.isdir(raw_path):
        raw_counts = count_images_by_class(raw_path)
        raw_total = sum(raw_counts.values())
        if raw_total == 0:
            print("   ⚠️  Vazio!")
            warnings.append("raw_dataset está vazio")
        else:
            for cls, cnt in sorted(raw_counts.items()):
                print(f"   {cls:15s}: {cnt:4d} imagens")
            print(f"   {'TOTAL':15s}: {raw_total:4d}")
    else:
        print("   ❌ Não encontrado!")
        warnings.append("raw_dataset/ não encontrado")

    # ---- DATASET PREPARADO ----
    print(f"\n📁 Dataset: {dataset_path}")
    if not os.path.isdir(dataset_path):
        print("   ❌ Não encontrado!")
        print("   💡 Execute: python src/prepare_dataset.py")
        warnings.append("dataset/ não encontrado")
    else:
        grand_total = 0
        expected_classes = {"normal", "defeituoso"}

        for split in ["train", "val", "test"]:
            split_path = os.path.join(dataset_path, split)
            print(f"\n   📂 {split}/")

            if not os.path.isdir(split_path):
                print(f"      ❌ Não encontrado!")
                warnings.append(f"{split}/ não encontrado")
                continue

            counts = count_images_by_class(split_path)
            split_total = sum(counts.values())
            grand_total += split_total

            if split_total == 0:
                print(f"      ⚠️  Vazio!")
                warnings.append(f"{split}/ está vazio")
                continue

            found_classes = set(counts.keys())

            for cls in sorted(expected_classes):
                cnt = counts.get(cls, 0)
                status = "✅" if cnt > 0 else "❌"
                print(f"      {status} {cls:15s}: {cnt:4d}")
                if cnt == 0:
                    warnings.append(f"{split}/{cls}/ sem imagens")
                elif cnt < 10 and split == "train":
                    warnings.append(f"{split}/{cls}/ tem poucas imagens ({cnt})")

            # Classes extras inesperadas
            extras = found_classes - expected_classes
            for cls in extras:
                print(f"      ❓ {cls:15s}: {counts[cls]:4d} (inesperada)")
                warnings.append(f"Classe inesperada: {split}/{cls}/")

            print(f"      {'TOTAL':18s}: {split_total:4d}")

        print(f"\n   📊 Total geral: {grand_total} imagens")

        # Verificar balanceamento no train
        train_path = os.path.join(dataset_path, "train")
        if os.path.isdir(train_path):
            tc = count_images_by_class(train_path)
            vals = [v for v in tc.values() if v > 0]
            if len(vals) >= 2:
                ratio = max(vals) / min(vals)
                if ratio > 3:
                    warnings.append(
                        f"Desequilíbrio no treino: {max(vals)} vs {min(vals)} "
                        f"({ratio:.1f}x)"
                    )
                elif ratio > 2:
                    warnings.append(
                        f"Desequilíbrio leve no treino: razão {ratio:.1f}x"
                    )

    # ---- RESUMO DE AVISOS ----
    if warnings:
        print(f"\n{'─' * 60}")
        print(f"   ⚠️  {len(warnings)} aviso(s):\n")
        for i, w in enumerate(warnings, 1):
            print(f"   {i}. {w}")
    else:
        print(f"\n   ✅ Tudo OK! Dataset pronto para treino.")

    print()


if __name__ == "__main__":
    check_dataset()
