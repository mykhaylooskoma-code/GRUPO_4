import matplotlib.pyplot as plt
from pathlib import Path

def _prep(outpath): Path(outpath).parent.mkdir(parents=True, exist_ok=True)

def save_gray(img2d, outpath, title=""):
    _prep(outpath); plt.figure(); plt.imshow(img2d, cmap="gray")
    plt.title(title); plt.axis("off"); plt.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close()

def save_heat(img2d, outpath, title=""):
    _prep(outpath); plt.figure(); plt.imshow(img2d)
    plt.title(title); plt.axis("off"); plt.colorbar(fraction=0.046, pad=0.04)
    plt.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close()
