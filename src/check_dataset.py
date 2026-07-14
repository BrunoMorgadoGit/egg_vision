"""
EggVision - Verificação do Dataset
====================================
Verifica a estrutura do dataset e exibe estatísticas detalhadas.

Uso:
  python src/check_dataset.py
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    IMBALANCE_WARNING_RATIO,
    MIN_TRAIN_IMAGES_PER_CLASS,
    MIN_VAL_IMAGES_PER_CLASS,
    PROJECT_CLASSES,
    REQUIRED_SPLITS,
    VALID_IMAGE_EXTENSIONS,
    collect_dataset_image_records,
    collect_invalid_files,
    count_images_by_class,
    find_cross_split_similar_images,
    get_active_training_classes,
    get_project_root,
    get_split_counts,
    scan_class_directory,
)


def relpath(path: str, root: str) -> str:
    """Formata caminhos relativos para mensagens mais curtas."""
    return os.path.relpath(path, root)


def print_invalid_files(invalid_files: list[tuple[str, str]],
                        root: str,
                        warnings: list[str],
                        label: str) -> None:
    """Mostra arquivos inválidos sem poluir a saída em datasets grandes."""
    if not invalid_files:
        return

    warnings.append(f"{label}: {len(invalid_files)} arquivo(s) inválido(s)")
    print(f"\n   ⚠️  Arquivos inválidos em {label}:")
    for filepath, reason in invalid_files[:10]:
        print(f"      - {relpath(filepath, root)} ({reason})")
    if len(invalid_files) > 10:
        print(f"      ... +{len(invalid_files) - 10} arquivo(s)")


def print_name_pattern_warnings(root: str, warnings: list[str]) -> None:
    """Avisa quando imagens não têm identificador ovoXX no nome."""
    raw_path = os.path.join(root, "raw_dataset")
    bad_names = []

    for class_name in PROJECT_CLASSES:
        class_path = os.path.join(raw_path, class_name)
        scan = scan_class_directory(class_path)
        for filepath in scan["images"]:
            filename = os.path.basename(filepath)
            if not re.search(r"ovo\d+", filename, re.IGNORECASE):
                bad_names.append(filepath)

    if not bad_names:
        return

    warnings.append(
        f"{len(bad_names)} imagem(ns) em raw_dataset sem padrão ovoXX no nome"
    )
    print("\n   ⚠️  Nomes sem identificador ovoXX:")
    for filepath in bad_names[:10]:
        print(f"      - {relpath(filepath, root)}")
    if len(bad_names) > 10:
        print(f"      ... +{len(bad_names) - 10} arquivo(s)")
    print(
        "      Sem ovoXX, cada arquivo será tratado como um ovo diferente no split."
    )


def check_raw_dataset(root: str, warnings: list[str]) -> None:
    """Verifica raw_dataset/normal|rachado|sujo."""
    raw_path = os.path.join(root, "raw_dataset")
    print(f"\n📁 Raw Dataset: {raw_path}")

    if not os.path.isdir(raw_path):
        print("   ❌ Não encontrado!")
        warnings.append("raw_dataset/ não encontrado")
        return

    raw_counts = count_images_by_class(raw_path)
    raw_total = 0

    for class_name in PROJECT_CLASSES:
        class_path = os.path.join(raw_path, class_name)
        if not os.path.isdir(class_path):
            print(f"   ❌ {class_name:10s}: pasta ausente")
            warnings.append(f"raw_dataset/{class_name}/ não encontrado")
            continue

        count = raw_counts.get(class_name, 0)
        raw_total += count
        status = "✅" if count > 0 else "⚠️ "
        print(f"   {status} {class_name:10s}: {count:4d} imagens")

    print(f"   {'TOTAL':13s}: {raw_total:4d}")

    if raw_total == 0:
        warnings.append("raw_dataset está vazio")
    if raw_counts.get("sujo", 0) == 0:
        warnings.append("raw_dataset/sujo/ está vazio; a classe sujo ainda não será treinada")

    invalid_files = collect_invalid_files(raw_path, splits=None)
    print_invalid_files(invalid_files, root, warnings, "raw_dataset")
    print_name_pattern_warnings(root, warnings)


def check_prepared_dataset(root: str, warnings: list[str]) -> None:
    """Verifica dataset/train|val|test/normal|rachado|sujo."""
    dataset_path = os.path.join(root, "dataset")
    print(f"\n📁 Dataset preparado: {dataset_path}")

    if not os.path.isdir(dataset_path):
        print("   ❌ Não encontrado!")
        print("   💡 Execute: python src/prepare_dataset.py")
        warnings.append("dataset/ não encontrado")
        return

    counts = get_split_counts(dataset_path)
    grand_total = 0

    for split in REQUIRED_SPLITS:
        split_path = os.path.join(dataset_path, split)
        print(f"\n   📂 {split}/")

        if not os.path.isdir(split_path):
            print("      ❌ Não encontrado!")
            warnings.append(f"dataset/{split}/ não encontrado")
            continue

        found_dirs = {
            name for name in os.listdir(split_path)
            if os.path.isdir(os.path.join(split_path, name))
        }
        unexpected = found_dirs - set(PROJECT_CLASSES)
        for class_name in sorted(unexpected):
            print(f"      ❓ {class_name:10s}: classe inesperada")
            warnings.append(f"Classe inesperada: dataset/{split}/{class_name}/")

        split_total = 0
        for class_name in PROJECT_CLASSES:
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                print(f"      ❌ {class_name:10s}: pasta ausente")
                warnings.append(f"dataset/{split}/{class_name}/ não encontrado")
                continue

            count = counts[split][class_name]
            split_total += count
            grand_total += count
            status = "✅" if count > 0 else "⚠️ "
            print(f"      {status} {class_name:10s}: {count:4d}")

            if count == 0 and class_name != "sujo":
                warnings.append(f"dataset/{split}/{class_name}/ sem imagens")
            elif count == 0 and class_name == "sujo":
                warnings.append(f"dataset/{split}/sujo/ vazio; ignorado temporariamente")

        print(f"      {'TOTAL':13s}: {split_total:4d}")

    print(f"\n   📊 Total geral: {grand_total} imagens")
    if grand_total == 0:
        warnings.append("dataset preparado está vazio")

    active_classes = get_active_training_classes(counts)
    if active_classes:
        print(f"   🏷️  Classes treináveis agora: {', '.join(active_classes)}")
    else:
        print("   🏷️  Classes treináveis agora: nenhuma")

    if "sujo" not in active_classes:
        print(
            "   ⚠️  'sujo' não será treinada agora. O modelo não reconhecerá ovos sujos "
            "até receber imagens reais em train/val/test e ser retreinado."
        )

    if len(active_classes) < 2:
        warnings.append("treinamento requer pelo menos duas classes com imagens em train e val")

    for class_name in active_classes:
        train_count = counts["train"][class_name]
        val_count = counts["val"][class_name]
        if train_count < MIN_TRAIN_IMAGES_PER_CLASS:
            warnings.append(
                f"dataset/train/{class_name}/ tem poucas imagens ({train_count}); "
                f"recomendado >= {MIN_TRAIN_IMAGES_PER_CLASS}"
            )
        if val_count < MIN_VAL_IMAGES_PER_CLASS:
            warnings.append(
                f"dataset/val/{class_name}/ tem poucas imagens ({val_count}); "
                f"recomendado >= {MIN_VAL_IMAGES_PER_CLASS}"
            )

    train_counts = [counts["train"][class_name] for class_name in active_classes]
    train_counts = [value for value in train_counts if value > 0]
    if len(train_counts) >= 2:
        ratio = max(train_counts) / min(train_counts)
        if ratio > IMBALANCE_WARNING_RATIO:
            warnings.append(
                f"Desequilíbrio alto no treino: {max(train_counts)} vs "
                f"{min(train_counts)} ({ratio:.1f}x)"
            )
        elif ratio > 2:
            warnings.append(f"Desequilíbrio leve no treino: razão {ratio:.1f}x")

    invalid_files = collect_invalid_files(dataset_path)
    print_invalid_files(invalid_files, root, warnings, "dataset")

    records = collect_dataset_image_records(dataset_path)
    exact_pairs, similar_pairs = find_cross_split_similar_images(records)

    if exact_pairs:
        warnings.append(
            f"{len(exact_pairs)} possível(is) duplicata(s) exata(s) entre splits"
        )
        print("\n   ⚠️  Possíveis duplicatas exatas entre splits:")
        for left, right in exact_pairs[:5]:
            print(f"      - {left['split']}/{left['class']} ↔ {right['split']}/{right['class']}")
            print(f"        {relpath(left['path'], root)}")
            print(f"        {relpath(right['path'], root)}")

    if similar_pairs:
        warnings.append(
            f"{len(similar_pairs)} imagem(ns) muito semelhante(s) entre splits"
        )
        print("\n   ⚠️  Possíveis imagens muito semelhantes entre splits:")
        for left, right, distance in similar_pairs[:5]:
            print(
                f"      - {left['split']}/{left['class']} ↔ "
                f"{right['split']}/{right['class']} (distância={distance})"
            )
            print(f"        {relpath(left['path'], root)}")
            print(f"        {relpath(right['path'], root)}")


def check_dataset():
    """Verifica e exibe estatísticas do raw_dataset e dataset."""
    print("=" * 60)
    print("   🥚 EggVision — Verificação do Dataset")
    print("=" * 60)
    print(f"   Extensões aceitas: {', '.join(VALID_IMAGE_EXTENSIONS)}")

    root = get_project_root()
    warnings = []

    check_raw_dataset(root, warnings)
    check_prepared_dataset(root, warnings)

    if warnings:
        print(f"\n{'─' * 60}")
        print(f"   ⚠️  {len(warnings)} aviso(s):\n")
        for index, warning in enumerate(warnings, 1):
            print(f"   {index}. {warning}")
    else:
        print("\n   ✅ Tudo OK! Dataset pronto para treino.")

    print()


if __name__ == "__main__":
    check_dataset()
