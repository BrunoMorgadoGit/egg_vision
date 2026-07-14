"""
EggVision - Preparação do Dataset
===================================
Organiza as imagens de raw_dataset/ em dataset/train, val e test.

Mapeamento:
  raw_dataset/normal/  → dataset/{split}/normal/
  raw_dataset/rachado/ → dataset/{split}/rachado/
  raw_dataset/sujo/    → dataset/{split}/sujo/

Uso:
  python src/prepare_dataset.py
  python src/prepare_dataset.py --train 0.7 --val 0.15 --test 0.15
"""

import os
import re
import sys
import argparse
import shutil
import random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    get_project_root, safe_mkdir,
    clear_directory, PROJECT_CLASSES, REQUIRED_SPLITS,
    ensure_dataset_structure, scan_class_directory,
)

RAW_TO_CLASS = {
    "normal": "normal",
    "rachado": "rachado",
    "sujo": "sujo",
}


def extract_egg_id(filename: str) -> str:
    """Extrai o ID do ovo (ex: 'ovo01') do nome do arquivo."""
    match = re.search(r"(ovo\d+)", filename, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return os.path.splitext(filename)[0]


def collect_images(raw_path: str) -> dict:
    """Coleta imagens do raw_dataset agrupadas por classe e ID do ovo."""
    images = defaultdict(lambda: defaultdict(list))
    total_found = 0
    total_skipped = 0

    for raw_folder, target_class in RAW_TO_CLASS.items():
        folder_path = os.path.join(raw_path, raw_folder)
        if not os.path.isdir(folder_path):
            print(f"   ⚠️  Pasta não encontrada: {folder_path}")
            continue

        scan = scan_class_directory(folder_path)
        skipped = len(scan["invalid"])
        count = 0
        for fp in scan["images"]:
            fn = os.path.basename(fp)
            egg_id = extract_egg_id(fn)
            group_key = f"{raw_folder}_{egg_id}"
            images[target_class][group_key].append(fp)
            count += 1

        total_found += count
        total_skipped += skipped
        skip_msg = f" ({skipped} ignoradas)" if skipped else ""
        print(f"   📂 {raw_folder}/ → {target_class}: {count} imagens{skip_msg}")

    print(f"\n   📊 Total válidas: {total_found}")
    if total_skipped:
        print(f"   ⚠️  Ignoradas: {total_skipped}")
    return dict(images)


def split_groups(groups, train_r, val_r, test_r, seed):
    """Divide grupos em train/val/test preservando integridade dos grupos."""
    keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    total = len(keys)
    n_train = max(1, round(total * train_r))
    n_val = max(1, round(total * val_r))

    if total == 1:
        n_train, n_val, n_test = 1, 0, 0
    elif total == 2:
        n_train, n_val, n_test = 1, 1, 0
    else:
        n_test = total - n_train - n_val
        if n_test < 1:
            n_test = 1
            n_train = total - n_val - n_test

    train_f = [f for k in keys[:n_train] for f in groups[k]]
    val_f = [f for k in keys[n_train:n_train+n_val] for f in groups[k]]
    test_f = [f for k in keys[n_train+n_val:] for f in groups[k]]
    return train_f, val_f, test_f


def copy_files(files, dest_dir):
    """Copia arquivos para o diretório de destino."""
    safe_mkdir(dest_dir)
    count = 0
    for fp in files:
        fn = os.path.basename(fp)
        dest = os.path.join(dest_dir, fn)
        if os.path.exists(dest):
            name, ext = os.path.splitext(fn)
            c = 1
            while os.path.exists(dest):
                dest = os.path.join(dest_dir, f"{name}_{c}{ext}")
                c += 1
        shutil.copy2(fp, dest)
        count += 1
    return count


def prepare_dataset(train_r=0.70, val_r=0.15, test_r=0.15, seed=42):
    """Organiza raw_dataset em dataset/train, val, test."""
    print("=" * 60)
    print("   🥚 EggVision — Preparação do Dataset")
    print("=" * 60)

    root = get_project_root()
    raw_path = os.path.join(root, "raw_dataset")
    dataset_path = os.path.join(root, "dataset")

    total_r = train_r + val_r + test_r
    if abs(total_r - 1.0) > 0.01:
        print(f"\n❌ Proporções devem somar 1.0! ({total_r})")
        sys.exit(1)

    print(f"\n📁 Raw: {raw_path}")
    print(f"📁 Dest: {dataset_path}")
    print(f"📊 Split: train={train_r:.0%} | val={val_r:.0%} | test={test_r:.0%}")
    print(f"🎲 Seed: {seed}")

    if not os.path.isdir(raw_path):
        print(f"\n❌ raw_dataset/ não encontrada em: {raw_path}")
        print("💡 Crie: raw_dataset/normal/, raw_dataset/rachado/, raw_dataset/sujo/")
        sys.exit(1)

    print(f"\n🔍 Coletando imagens...")
    images = collect_images(raw_path)

    print(f"\n🧹 Limpando dataset/...")
    clear_directory(dataset_path)
    ensure_dataset_structure(dataset_path)

    if not images:
        print("\n⚠️  Nenhuma imagem válida encontrada.")
        print("   A estrutura dataset/train|val|test/normal|rachado|sujo foi criada vazia.")
        print("   Adicione imagens reais em raw_dataset/ e execute novamente.")
        return {"train": {}, "val": {}, "test": {}}

    print(f"\n📦 Organizando...\n")
    stats = {"train": {}, "val": {}, "test": {}}

    for cls in PROJECT_CLASSES:
        groups = images.get(cls, {})
        total_imgs = sum(len(f) for f in groups.values())

        if total_imgs == 0:
            print(f"   ⚠️  {cls}: 0 imagens (pasta criada vazia)")
            for split_name in REQUIRED_SPLITS:
                stats[split_name][cls] = 0
            continue

        print(f"   📂 {cls}: {len(groups)} grupos, {total_imgs} imagens")

        train_f, val_f, test_f = split_groups(groups, train_r, val_r, test_r, seed)

        for split_name, files in [("train",train_f),("val",val_f),("test",test_f)]:
            dest = os.path.join(dataset_path, split_name, cls)
            cnt = copy_files(files, dest)
            stats[split_name][cls] = cnt
            print(f"      → {split_name}: {cnt}")
        print()

    # Resumo
    print("=" * 60)
    print("   ✅ Dataset preparado!")
    print("=" * 60)

    total = 0
    for sp in ["train", "val", "test"]:
        sp_total = sum(stats[sp].values())
        total += sp_total
        cls_str = " | ".join(f"{c}: {n}" for c, n in sorted(stats[sp].items()))
        print(f"   {sp:5s}: {sp_total:4d}  ({cls_str})")
    print(f"   total: {total:4d}")

    # Avisos
    for cls in PROJECT_CLASSES:
        tc = stats["train"].get(cls, 0)
        if tc == 0 and cls == "sujo":
            print(
                "\n   ⚠️  'sujo' está vazio. A classe foi preservada na estrutura, "
                "mas não será treinada enquanto não houver imagens reais."
            )
        elif tc < 20:
            print(f"\n   ⚠️  '{cls}' tem apenas {tc} imgs de treino.")

    tc = [v for v in stats["train"].values() if v > 0]
    if len(tc) >= 2 and min(tc) > 0 and max(tc)/min(tc) > 3:
        print(f"\n   ⚠️  Desequilíbrio: {max(tc)} vs {min(tc)} ({max(tc)/min(tc):.1f}x)")

    print(f"\n💡 Próximo: python src/check_dataset.py")


def main():
    parser = argparse.ArgumentParser(description="EggVision - Preparação do Dataset")
    parser.add_argument("--train", type=float, default=0.70, help="Proporção treino")
    parser.add_argument("--val", type=float, default=0.15, help="Proporção validação")
    parser.add_argument("--test", type=float, default=0.15, help="Proporção teste")
    parser.add_argument("--seed", type=int, default=42, help="Seed (padrão: 42)")
    args = parser.parse_args()
    prepare_dataset(args.train, args.val, args.test, args.seed)


if __name__ == "__main__":
    main()
