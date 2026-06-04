"""
prepare_vsl400_labels.py

Tạo label CSV cho VSL400 theo format cần thiết:
- Per-view CSVs (front/left/right): file_name, label_id
- Three-view CSV: front, left, right, label_id

Cấu trúc VSL400:
  merged_dataset/front_view/000000.mp4
  merged_dataset/left_view/000000.mp4
  merged_dataset/right_view/000000.mp4
  train_labels_clean.csv: sample_id, label_id, gloss

Output (tạo tại VSL400_LABEL_DIR):
  front/train_labels.csv, front/val_labels.csv, front/test_labels.csv
  left/train_labels.csv,  left/val_labels.csv,  left/test_labels.csv
  right/train_labels.csv, right/val_labels.csv, right/test_labels.csv
  three_view/train_labels.csv, three_view/val_labels.csv, three_view/test_labels.csv
"""

import os
import pandas as pd

# ─── CẤU HÌNH ────────────────────────────────────────────────────
VSL400_ROOT   = '/media/ibmelab/ibme31/vsl400_dataset'
VSL400_LABEL_DIR = '/media/ibmelab/ibme31/vsl400_dataset/labels'

VIEW_MAP = {
    'front': 'front_view',
    'left':  'left_view',
    'right': 'right_view',
}
SPLITS = ['train', 'val', 'test']
SPLIT_CSV = {
    'train': 'train_labels_clean.csv',
    'val':   'val_labels_clean.csv',
    'test':  'test_labels_clean.csv',
}
# ─────────────────────────────────────────────────────────────────


def make_per_view_csv(df, view_name, view_dir_name, out_dir, split):
    """
    Tạo CSV format: file_name, label_id
    file_name = f"{view_dir_name}/{sample_id:06d}.mp4"
    """
    rows = []
    missing = 0
    for _, row in df.iterrows():
        sample_id = str(row['sample_id']).zfill(6)
        fname = f"{view_dir_name}/{sample_id}.mp4"
        # Kiểm tra file tồn tại
        full_path = os.path.join(VSL400_ROOT, 'merged_dataset', fname)
        if not os.path.exists(full_path):
            missing += 1
        rows.append({'file_name': fname, 'label_id': int(row['label_id'])})

    out_path = os.path.join(out_dir, view_name, f'{split}_labels.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"  [{split}/{view_name}] {len(out_df)} samples, {missing} missing files -> {out_path}")
    return out_df


def make_three_view_csv(df, out_dir, split):
    """
    Tạo CSV format: front, left, right, label_id
    Tương tự labelThreeView của VSL200
    """
    rows = []
    for _, row in df.iterrows():
        sample_id = str(row['sample_id']).zfill(6)
        rows.append({
            'front': f"front_view/{sample_id}.mp4",
            'left':  f"left_view/{sample_id}.mp4",
            'right': f"right_view/{sample_id}.mp4",
            'label_id': int(row['label_id']),
        })

    out_path = os.path.join(out_dir, 'three_view', f'{split}_labels.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"  [{split}/three_view] {len(out_df)} samples -> {out_path}")
    return out_df


def main():
    print("=== Preparing VSL400 label CSVs ===\n")

    for split in SPLITS:
        csv_path = os.path.join(VSL400_ROOT, SPLIT_CSV[split])
        df = pd.read_csv(csv_path)
        print(f"[{split}] {len(df)} samples loaded")

        # Per-view CSVs
        for view_name, view_dir in VIEW_MAP.items():
            make_per_view_csv(df, view_name, view_dir, VSL400_LABEL_DIR, split)

        # Three-view CSV
        make_three_view_csv(df, VSL400_LABEL_DIR, split)
        print()

    # In thống kê
    print("=== Label statistics ===")
    for split in SPLITS:
        df = pd.read_csv(os.path.join(VSL400_ROOT, SPLIT_CSV[split]))
        n_classes = df['label_id'].nunique()
        print(f"  {split}: {len(df)} samples, {n_classes} classes, "
              f"label range [{df['label_id'].min()}, {df['label_id'].max()}]")

    print("\nDone! Label CSVs created at:", VSL400_LABEL_DIR)


if __name__ == '__main__':
    main()