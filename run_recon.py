# Inspired by matches the Gradio App file.
# Usage:
#   python run_recon.py --images /path/to/images --out results
#   python run_recon.py --images /path/to/images --out results --views 15
#   python run_recon.py --images /path/to/images --out results --apache --no-mesh

import argparse
import os
import time

import numpy as np
import torch

if torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from mapanything.models import MapAnything
from mapanything.utils.geometry import depthmap_to_world_frame
from mapanything.utils.hf_utils.viz import predictions_to_glb
from mapanything.utils.image import load_images


def parse_args():
    p = argparse.ArgumentParser(description="Headless MapAnything reconstruction -> GLB + depth manifest")
    p.add_argument("--images", required=True, help="Folder of input images (or a single image)")
    p.add_argument("--out", default="recon_output", help="Output directory")
    p.add_argument("--views", type=int, default=0,
                   help="If >0, evenly subsample to this many views")
    p.add_argument("--apache", action="store_true",
                   help="Use the Apache-2.0 model (facebook/map-anything-apache) instead of the default CC-BY-NC.")
    p.add_argument("--memory-efficient", action="store_true",
                   help="Trade speed for far lower peak memory (needed for high view counts / small GPUs).")
    p.add_argument("--no-mask", action="store_true", help="Disable the non-ambiguous output mask.")
    p.add_argument("--no-mesh", action="store_true", help="Export point cloud instead of a mesh.")
    p.add_argument("--conf-percentile", type=float, default=10.0,
                   help="Confidence percentile threshold for GLB filtering (default 10).")
    p.add_argument("--save-depth-npy", action="store_true",
                   help="Also save each per-view depth map as a .npy file (raw metric depth).")
    return p.parse_args()


def subsample_paths(image_dir, n):
    """Return an evenly-spaced subset of n image paths from a folder."""
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    files = sorted(
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(exts)
    )
    if n <= 0 or n >= len(files):
        return files, files
    idx = np.linspace(0, len(files) - 1, n).round().astype(int)
    idx = sorted(set(idx.tolist()))
    return [files[i] for i in idx], files


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Resolve input images (with optional subsampling) ----
    if os.path.isdir(args.images):
        selected, all_files = subsample_paths(args.images, args.views)
        image_input = selected
        print(f"Found {len(all_files)} images; using {len(selected)}"
              + (f" (subsampled to {args.views})" if args.views else " (all)"))
    else:
        image_input = args.images  # single file or path string
        print(f"Using image path: {args.images}")

    # ---- Load model ----
    model_name = "facebook/map-anything-apache" if args.apache else "facebook/map-anything"
    print(f"Loading model {model_name} ...")
    model = MapAnything.from_pretrained(model_name).to(device)
    model.eval()

    # ---- Load images ----
    print("Loading images...")
    views = load_images(image_input)
    print(f"Loaded {len(views)} views")
    if len(views) == 0:
        raise ValueError("No images loaded. Check --images path.")

    # ---- Inference ----
    print("Running inference...")
    t0 = time.time()
    outputs = model.infer(
        views,
        apply_mask=(not args.no_mask),
        mask_edges=True,
        memory_efficient_inference=args.memory_efficient,
    )
    print(f"Inference done in {time.time() - t0:.2f}s")

    # ---- Build predictions dict (mirrors gradio_app.py run_model, lines ~147-204) ----
    extrinsic_list, intrinsic_list, world_points_list = [], [], []
    depth_maps_list, images_list, final_mask_list, confidences = [], [], [], []

    for pred in outputs:
        depthmap_torch = pred["depth_z"][0].squeeze(-1)        # (H, W)
        intrinsics_torch = pred["intrinsics"][0]               # (3, 3)
        camera_pose_torch = pred["camera_poses"][0]            # (4, 4)
        conf = pred["conf"][0].squeeze(-1)                     # (H, W)

        pts3d_computed, valid_mask = depthmap_to_world_frame(
            depthmap_torch, intrinsics_torch, camera_pose_torch
        )

        if "mask" in pred:
            mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        else:
            mask = np.ones_like(depthmap_torch.cpu().numpy(), dtype=bool)
        mask = mask & valid_mask.cpu().numpy()

        image = pred["img_no_norm"][0].cpu().numpy()

        extrinsic_list.append(camera_pose_torch.cpu().numpy())
        intrinsic_list.append(intrinsics_torch.cpu().numpy())
        world_points_list.append(pts3d_computed.cpu().numpy())
        depth_maps_list.append(depthmap_torch.cpu().numpy())
        images_list.append(image)
        final_mask_list.append(mask)
        confidences.append(conf.cpu().numpy())

    predictions = {}
    predictions["extrinsic"] = np.stack(extrinsic_list, axis=0)        # (S, 4, 4)
    predictions["intrinsic"] = np.stack(intrinsic_list, axis=0)        # (S, 3, 3)
    predictions["world_points"] = np.stack(world_points_list, axis=0)  # (S, H, W, 3)
    predictions["conf"] = np.stack(confidences, axis=0)                # (S, H, W)

    depth_maps = np.stack(depth_maps_list, axis=0)
    if len(depth_maps.shape) == 3:
        depth_maps = depth_maps[..., np.newaxis]                       # (S, H, W, 1)
    predictions["depth"] = depth_maps
    predictions["images"] = np.stack(images_list, axis=0)              # (S, H, W, 3)
    predictions["final_mask"] = np.stack(final_mask_list, axis=0)      # (S, H, W)

    # ---- Save raw predictions ----
    npz_path = os.path.join(args.out, "predictions.npz")
    np.savez(npz_path, **predictions)
    print(f"Saved raw predictions: {npz_path}")

    # ---- Export GLB ----
    glb_path = os.path.join(args.out, "reconstruction.glb")
    glbscene = predictions_to_glb(
        predictions,
        filter_by_frames="All",
        show_cam=True,
        mask_black_bg=False,
        mask_white_bg=False,
        as_mesh=(not args.no_mesh),
        conf_percentile=args.conf_percentile,
    )
    glbscene.export(file_obj=glb_path)
    print(f"Exported GLB: {glb_path}")

    # ---- save per-view depth maps as colorized PNGs + raw .npy ----
    depth_dir = os.path.join(args.out, "depth_maps")
    os.makedirs(depth_dir, exist_ok=True)
    try:
        import cv2
        have_cv2 = True
    except ImportError:
        have_cv2 = False

    depth_png_paths = []
    for i in range(predictions["depth"].shape[0]):
        d = predictions["depth"][i, ..., 0]              # (H, W) metric depth
        m = predictions["final_mask"][i]                 # (H, W) bool
        png_path = os.path.join(depth_dir, f"view_{i:03d}_depth.png")
        if have_cv2:
            valid = d[m] if m.any() else d.reshape(-1)
            lo = float(np.percentile(valid, 2)) if valid.size else 0.0
            hi = float(np.percentile(valid, 98)) if valid.size else 1.0
            norm = np.clip((d - lo) / (hi - lo + 1e-8), 0, 1)
            vis = (norm * 255).astype(np.uint8)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
            vis[~m] = 0  # black out masked/invalid pixels
            cv2.imwrite(png_path, vis)
            depth_png_paths.append(png_path)
        if args.save_depth_npy:
            np.save(os.path.join(depth_dir, f"view_{i:03d}_depth.npy"), d)

    # ---- Write the .txt manifest: images used + depth-map info ----
    manifest_path = os.path.join(args.out, "manifest.txt")
    S = predictions["depth"].shape[0]
    with open(manifest_path, "w") as f:
        f.write("MapAnything Reconstruction Manifest\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp        : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model            : {model_name}\n")
        f.write(f"Device           : {device}"
                + (f" ({torch.cuda.get_device_name(0)})" if device == 'cuda' else "") + "\n")
        f.write(f"Views used       : {S}\n")
        f.write(f"GLB output       : {os.path.abspath(glb_path)}\n")
        f.write(f"Raw predictions  : {os.path.abspath(npz_path)}\n")
        f.write(f"Depth maps dir   : {os.path.abspath(depth_dir)}\n")
        f.write("=" * 60 + "\n\n")

        # If we loaded from a folder, list the exact source files used.
        if isinstance(image_input, list):
            f.write("Source images used (in order):\n")
            for i, p in enumerate(image_input):
                f.write(f"  [{i:03d}] {os.path.abspath(p)}\n")
            f.write("\n")

        f.write("Per-view depth-map summary (metric units):\n")
        f.write(f"{'view':>5} | {'depth_min':>9} | {'depth_max':>9} | "
                f"{'depth_mean':>10} | {'valid_px%':>9} | depth_png\n")
        f.write("-" * 80 + "\n")
        for i in range(S):
            d = predictions["depth"][i, ..., 0]
            m = predictions["final_mask"][i]
            valid = d[m] if m.any() else d.reshape(-1)
            dmin = float(valid.min()) if valid.size else float("nan")
            dmax = float(valid.max()) if valid.size else float("nan")
            dmean = float(valid.mean()) if valid.size else float("nan")
            valid_pct = 100.0 * float(m.mean())
            png_name = (os.path.basename(depth_png_paths[i])
                        if i < len(depth_png_paths) else "(cv2 not installed)")
            f.write(f"{i:>5} | {dmin:9.3f} | {dmax:9.3f} | {dmean:10.3f} | "
                    f"{valid_pct:9.2f} | {png_name}\n")

    print(f"Wrote manifest: {manifest_path}")
    print("\nDone. Outputs:")
    print(f"  GLB       : {glb_path}")
    print(f"  Manifest  : {manifest_path}")
    print(f"  Depth maps: {depth_dir}/")


if __name__ == "__main__":
    main()