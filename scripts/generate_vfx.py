"""Generate VFX overlay images for Director Factory.
Each overlay is a 1024x576 PNG with alpha channel suitable for ffmpeg overlay.
"""
import numpy as np
from PIL import Image
from pathlib import Path

VFX_DIR = Path(__file__).parent.parent / "assets" / "vfx"
W, H = 1024, 576


def save_overlay(path: Path, rgba: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(rgba, "RGBA")
    img.save(path, "PNG")


def create_canvas() -> np.ndarray:
    return np.zeros((H, W, 4), dtype=np.uint8)


# ── VFX Generators ──────────────────────────────────────

def gen_film_grain():
    """Film grain: random semi-transparent noise across the frame."""
    rng = np.random.default_rng()
    canvas = create_canvas()
    v = rng.integers(0, 31, (H, W), dtype=np.uint8)
    a = rng.integers(15, 41, (H, W), dtype=np.uint8)
    canvas[:, :, 0] = v
    canvas[:, :, 1] = v
    canvas[:, :, 2] = v
    canvas[:, :, 3] = a
    save_overlay(VFX_DIR / "film_grain_overlay.png", canvas)
    print("  film_grain_overlay.png")


def gen_vignette_dark():
    """Vignette: darker edges, center is clear."""
    canvas = create_canvas()
    cx, cy = W / 2, H / 2
    max_dist = np.sqrt(cx**2 + cy**2)
    ys = np.arange(H).reshape(-1, 1)
    xs = np.arange(W).reshape(1, -1)
    dist = np.sqrt((xs - cx)**2 + (ys - cy)**2) / max_dist
    alpha = np.clip((dist - 0.3) * 350, 0, 180).astype(np.uint8)
    canvas[:, :, 3] = alpha
    save_overlay(VFX_DIR / "vignette_dark_overlay.png", canvas)
    print("  vignette_dark_overlay.png")


def gen_light_leak():
    """Light leak: warm orange/yellow gradient from top-left corner."""
    canvas = create_canvas()
    ys = np.arange(H).reshape(-1, 1) / H
    xs = np.arange(W).reshape(1, -1) / W
    dist = np.sqrt(xs**2 + ys**2)
    alpha = np.clip((1 - dist * 1.5) * 100, 0, 255).astype(np.uint8)
    mask = alpha > 0
    canvas[:, :, 0] = np.where(mask, np.clip(255 * alpha / 100, 0, 255), 0).astype(np.uint8)
    canvas[:, :, 1] = np.where(mask, np.clip(180 * alpha / 100, 0, 255), 0).astype(np.uint8)
    canvas[:, :, 2] = np.where(mask, np.clip(60 * alpha / 100, 0, 255), 0).astype(np.uint8)
    canvas[:, :, 3] = alpha
    save_overlay(VFX_DIR / "light_leak_overlay.png", canvas)
    print("  light_leak_overlay.png")


def gen_dust_particles():
    """Dust particles: sparse small white dots floating."""
    canvas = create_canvas()
    rng = np.random.default_rng(42)
    ys = np.arange(H).reshape(-1, 1)
    xs = np.arange(W).reshape(1, -1)
    for _ in range(200):
        cx = rng.integers(0, W)
        cy = rng.integers(0, H)
        size = rng.integers(1, 5)
        alpha = rng.integers(60, 181)
        dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)
        mask = dist <= size
        fade = 1 - dist / (size + 1)
        a = (alpha * fade).astype(np.uint8)
        a[~mask] = 0
        # Blend: only overwrite where alpha is higher
        existing = canvas[:, :, 3]
        update = a > existing
        canvas[:, :, 0] = np.where(update, 255, canvas[:, :, 0])
        canvas[:, :, 1] = np.where(update, 255, canvas[:, :, 1])
        canvas[:, :, 2] = np.where(update, 240, canvas[:, :, 2])
        canvas[:, :, 3] = np.where(update, a, existing)
    save_overlay(VFX_DIR / "dust_particles_overlay.png", canvas)
    print("  dust_particles_overlay.png")


def gen_subtle_blur():
    """Subtle blur mask: very faint uniform white overlay for softening effect."""
    canvas = create_canvas()
    canvas[:, :, 0] = 255
    canvas[:, :, 1] = 255
    canvas[:, :, 2] = 255
    canvas[:, :, 3] = 15
    save_overlay(VFX_DIR / "subtle_blur_overlay.png", canvas)
    print("  subtle_blur_overlay.png")


if __name__ == "__main__":
    print("Generating VFX overlays...")
    gen_film_grain()
    gen_vignette_dark()
    gen_light_leak()
    gen_dust_particles()
    gen_subtle_blur()
    print(f"\nDone! 5 VFX overlays generated in {VFX_DIR}")
