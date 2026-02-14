#!/usr/bin/env python3
import argparse
import glob
import os
import re
import sys

import numpy as np
import tifffile
from PIL import Image


def reconstruct_image_from_patches_JUST_BIT(
    metadata_path,
    patch_folder,
    img_num,
    return_as_array=False,
    patch_index=None,
    base_key=None,
):
    """
    Reconstruct the original image using metadata coordinates (x, y, width, height),
    loading patches from patch_folder by basename pattern:
      *img=<img_num>_P=<P>.(png|tif|tiff)  or  *img_<img_num>_P=<P>.(png|tif|tiff)

    The metadata filenames are ignored except for extracting P numbers and coordinates.

    Parameters:
        metadata_path (str): Path to metadata file.
        patch_folder (str): Folder containing patch images.
        img_num (int): Image number to reconstruct (e.g., 23 -> use img=23_P=... or img_23_P=...).
        return_as_array (bool): If True, return NumPy array; else return PIL.Image.

    Returns:
        np.ndarray or PIL.Image.Image
    """
    # Read metadata file
    with open(metadata_path, 'r') as f:
        lines = f.read().splitlines()

    # Parse original size (width, height) from header
    w = h = None
    for line in lines:
        if "Original size:" in line:
            mw = re.search(r'width=(\d+)', line)
            mh = re.search(r'height=(\d+)', line)
            if mw and mh:
                w = int(mw.group(1)); h = int(mh.group(1))
            break
    if w is None or h is None:
        raise ValueError("Failed to parse Original size (width/height) from metadata.")

    # Collect entries: (P, x, y, pw, ph) from non-comment lines
    entries = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split(',')
        if len(parts) < 5:
            continue
        fname, x, y, pw, ph = parts[0], int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
        mP = re.search(r'P=(\d+)', fname)
        if not mP:
            continue
        P = int(mP.group(1))
        entries.append((P, x, y, pw, ph))
    if not entries:
        raise ValueError("No patch entries found in metadata (no P=... lines).")

    # Helper: find actual file for given img_num and P.
    # If an index is provided, use it to avoid cross-stack mixups.
    def find_patch_file(folder, img_num, P):
        if patch_index is not None and base_key is not None:
            return patch_index.get((base_key, img_num, P))
        patterns = [
            f'*img={img_num}_P={P}',
            f'*img_{img_num}_P={P}',
        ]
        for pat in patterns:
            for ext in ('png', 'tif', 'tiff'):
                matches = glob.glob(os.path.join(folder, pat + f'.{ext}'))
                if matches:
                    return matches[0]
        # Fallback: any extension if present
        for pat in patterns:
            matches = glob.glob(os.path.join(folder, pat + '.*'))
            if matches:
                return matches[0]
        return None

    # Helper: load raw patch as numpy array
    def load_raw(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.tif', '.tiff'):
            return tifffile.imread(path)
        if ext == '.png':
            with Image.open(path) as im:
                return np.array(im)
        raise ValueError(f"Unsupported patch extension: {ext}")

    # Determine mode (color vs gray) from the first available patch
    first_path = None
    for P, _, _, _, _ in sorted(entries):
        pth = find_patch_file(patch_folder, img_num, P)
        if pth:
            first_path = pth
            break
    if not first_path:
        raise FileNotFoundError(f"No patch files found for img={img_num} in {patch_folder}.")
    first_arr = load_raw(first_path)
    mode = 'color' if (first_arr.ndim == 3 and first_arr.shape[-1] >= 3) else 'gray'

    # Prepare canvas and weight map
    canvas = np.zeros((h, w, 3), dtype=np.float32) if mode == 'color' else np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)

    def prepare_patch(raw):
        # Convert raw array to chosen mode
        if raw.ndim == 2:
            gray = raw.astype(np.float32)
            return np.stack([gray, gray, gray], axis=-1) if mode == 'color' else gray
        if raw.ndim == 3:
            c = raw.shape[-1]
            if c == 1:
                gray = raw[..., 0].astype(np.float32)
                return np.stack([gray, gray, gray], axis=-1) if mode == 'color' else gray
            if c >= 3:
                rgb = raw[..., :3].astype(np.float32)
                if mode == 'gray':
                    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
                return rgb
        squeezed = np.squeeze(raw)
        if squeezed.ndim not in (2, 3):
            raise ValueError(f"Unsupported patch shape {raw.shape}")
        return prepare_patch(squeezed)

    used = 0
    missing = []
    for P, x, y, pw, ph in entries:
        patch_path = find_patch_file(patch_folder, img_num, P)
        if patch_path is None:
            missing.append(P)
            continue
        raw = load_raw(patch_path)
        patch = prepare_patch(raw)

        H, Wp = (patch.shape[:2] if mode == 'color' else patch.shape)
        if H < ph or Wp < pw:
            raise ValueError(f"Patch size {patch.shape} smaller than metadata ({ph}, {pw}) for {patch_path}")
        if H != ph or Wp != pw:
            patch = (patch[:ph, :pw, :] if mode == 'color' else patch[:ph, :pw])

        if mode == 'color':
            canvas[y:y+ph, x:x+pw, :] += patch
        else:
            canvas[y:y+ph, x:x+pw] += patch
        weight_map[y:y+ph, x:x+pw] += 1
        used += 1

    if missing:
        raise FileNotFoundError(f"Missing patches for img={img_num}, P values: {missing}")
    if used == 0:
        raise ValueError(f"No patches were used for reconstruction for img={img_num}.")

    # Normalize and finalize
    if mode == 'color':
        reconstructed = canvas / weight_map[..., None]
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        return reconstructed if return_as_array else Image.fromarray(reconstructed, mode='RGB')

    reconstructed = canvas / weight_map
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    return reconstructed if return_as_array else Image.fromarray(reconstructed, mode='L')


def extract_base_name(txt_path):
    """
    Extract the core identifier from a .txt metadata path.

    Example:
        crypts_Duodenum_IM0012_cropped_BIT_40X_frame30-61_patches_stitch_metadata.txt
        -> crypts_Duodenum_IM0012_cropped_BIT_40X_frame30-61
    """
    filename = os.path.basename(txt_path)
    match = re.match(r'(.*?)_patches_stitch_metadata\.txt', filename)
    return match.group(1) if match else None


def normalize_base_from_filename(fname):
    """
    Extract the base handle from a patch filename.
    Example:
      MUSE_BIT_crypts_..._img=0_P=1.tif -> crypts_... (lowercased for matching)
    """
    m = re.search(r'(crypts_.*)_img[=_]\d+_P=\d+', fname, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m = re.search(r'^(.*)_img[=_]\d+_P=\d+', fname, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return None


def build_patch_index(folder):
    """
    Build an index: (base_lower, img_num, P) -> filepath
    Also returns base_lower -> set(img_num) for fast lookup.
    """
    def base_variants(base_lower):
        # Allow matching when patch filenames carry a prefix not present in metadata.
        variants = {base_lower}
        for prefix in ("muse_bit_", "musebit_", "muse_"):
            if base_lower.startswith(prefix):
                variants.add(base_lower[len(prefix):])
        return variants

    index = {}
    base_to_imgs = {}
    if not os.path.isdir(folder):
        return index, base_to_imgs
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            continue
        m = re.search(r'img[=_](\d+)', fname)
        n = re.search(r'P=(\d+)', fname)
        if not m or not n:
            continue
        base = normalize_base_from_filename(fname)
        if not base:
            continue
        base = base.lower()
        img_num = int(m.group(1))
        p_num = int(n.group(1))
        for key in base_variants(base):
            index[(key, img_num, p_num)] = path
            base_to_imgs.setdefault(key, set()).add(img_num)
    return index, base_to_imgs


def reconstruct_all_stacks(bit_folder, vhe_folder, metadata_glob, bit_out_name, vhe_out_name, bit_ext, vhe_ext):
    bit_out_dir = os.path.join(bit_folder, bit_out_name)
    vhe_out_dir = os.path.join(vhe_folder, vhe_out_name)
    os.makedirs(bit_out_dir, exist_ok=True)
    os.makedirs(vhe_out_dir, exist_ok=True)

    metadata_paths = sorted(glob.glob(os.path.join(bit_folder, metadata_glob)))
    if not metadata_paths:
        raise FileNotFoundError(f"No metadata files found in {bit_folder} with {metadata_glob}")

    bit_index, bit_base_to_imgs = build_patch_index(bit_folder)
    vhe_index, vhe_base_to_imgs = build_patch_index(vhe_folder)

    summary = []
    for meta_path in metadata_paths:
        base = extract_base_name(meta_path)
        if not base:
            continue
        base_key = base.lower()

        bit_indices = bit_base_to_imgs.get(base_key, set())
        vhe_indices = vhe_base_to_imgs.get(base_key, set())

        # Reconstruct BIT images
        for img_num in sorted(bit_indices):
            recon = reconstruct_image_from_patches_JUST_BIT(
                metadata_path=meta_path,
                patch_folder=bit_folder,
                img_num=img_num,
                return_as_array=False,
                patch_index=bit_index,
                base_key=base_key,
            )
            out_name = f"{base}_img={img_num}.{bit_ext}"
            recon.save(os.path.join(bit_out_dir, out_name))

        # Reconstruct vHE images
        for img_num in sorted(vhe_indices):
            recon = reconstruct_image_from_patches_JUST_BIT(
                metadata_path=meta_path,
                patch_folder=vhe_folder,
                img_num=img_num,
                return_as_array=False,
                patch_index=vhe_index,
                base_key=base_key,
            )
            out_name = f"{base}_img={img_num}.{vhe_ext}"
            recon.save(os.path.join(vhe_out_dir, out_name))

        summary.append({
            "metadata": meta_path,
            "base": base,
            "bit_imgs": sorted(bit_indices),
            "vhe_imgs": sorted(vhe_indices),
        })

    return summary


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Reconstruct original FOVs from BIT/vHE patch folders and metadata."
    )
    parser.add_argument("--bit-folder", required=True, help="Folder containing BIT patches + metadata TXT files")
    parser.add_argument("--vhe-folder", required=True, help="Folder containing vHE patches")
    parser.add_argument(
        "--metadata-glob",
        default="*_patches_stitch_metadata.txt",
        help="Glob pattern for metadata files in BIT folder",
    )
    parser.add_argument("--bit-out-name", default="BIT_reconstructed", help="Output folder name inside BIT folder")
    parser.add_argument("--vhe-out-name", default="vHE_reconstructed", help="Output folder name inside vHE folder")
    parser.add_argument("--bit-ext", default="tif", help="Output extension for BIT reconstructed images")
    parser.add_argument("--vhe-ext", default="png", help="Output extension for vHE reconstructed images")
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    summary = reconstruct_all_stacks(
        bit_folder=args.bit_folder,
        vhe_folder=args.vhe_folder,
        metadata_glob=args.metadata_glob,
        bit_out_name=args.bit_out_name,
        vhe_out_name=args.vhe_out_name,
        bit_ext=args.bit_ext,
        vhe_ext=args.vhe_ext,
    )
    print(f"Reconstructed {len(summary)} stacks.")


if __name__ == "__main__":
    main(sys.argv[1:])
