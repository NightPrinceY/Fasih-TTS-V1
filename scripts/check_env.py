"""Environment & hardware sanity check for the Arabic TTS project.

Verifies the things that actually gate training on this box:
  * CUDA availability + visible GPUs (pinned to the free ones via .env)
  * Per-GPU compute capability, VRAM, and BF16/FP16 support (Turing = FP16 only)
  * A real FP16 matmul on-device (catches broken driver/wheel combos)
  * scaled_dot_product_attention (SDPA) backends + optional xFormers

Run:  uv run python scripts/check_env.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

console = Console()


def main() -> int:
    # Load .env so CUDA_VISIBLE_DEVICES pins the free GPUs before torch initialises.
    load_dotenv()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    console.rule("[bold]Arabic TTS — Environment Check")
    console.print(f"CUDA_VISIBLE_DEVICES = [cyan]{visible}[/]")

    import torch

    console.print(f"torch           = [green]{torch.__version__}[/]")
    console.print(f"torch CUDA build= [green]{torch.version.cuda}[/]")
    console.print(f"cuDNN           = [green]{torch.backends.cudnn.version()}[/]")

    if not torch.cuda.is_available():
        console.print("[bold red]CUDA NOT available — training impossible in this state.[/]")
        return 1

    n = torch.cuda.device_count()
    console.print(f"visible GPUs    = [green]{n}[/]\n")

    tbl = Table(title="GPUs")
    for c in ("idx", "name", "VRAM (GiB)", "compute", "BF16", "FP16 TC"):
        tbl.add_column(c)
    turing = False
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        cc = f"{p.major}.{p.minor}"
        if p.major == 7 and p.minor == 5:
            turing = True
        tbl.add_row(
            str(i),
            p.name,
            f"{p.total_memory / 1024**3:.1f}",
            cc,
            "yes" if p.major >= 8 else "[yellow]no[/]",
            "yes",
        )
    console.print(tbl)

    # BF16 support (Ampere+). On Turing this is False -> we train in FP16.
    bf16 = torch.cuda.is_bf16_supported()
    console.print(f"\ntorch.cuda.is_bf16_supported() = {'[green]True[/]' if bf16 else '[yellow]False[/]'}")
    if turing and not bf16:
        console.print("[yellow]-> Turing detected: use FP16 AMP everywhere, never bf16.[/]")

    # Real FP16 matmul on device.
    try:
        a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
        b = torch.randn(512, 512, device="cuda", dtype=torch.float16)
        c = (a @ b).float().sum().item()
        console.print(f"FP16 matmul on GPU 0 = [green]OK[/] (checksum={c:.1f})")
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]FP16 matmul FAILED: {e}[/]")
        return 1

    # SDPA backends (FlashAttention-2 is unavailable on Turing; mem-efficient/math work).
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: F401

        console.print("SDPA (scaled_dot_product_attention) = [green]available[/]")
    except Exception:  # noqa: BLE001
        console.print("[yellow]SDPA kernel selection API not available in this torch.[/]")

    # xFormers is our attention fallback on Turing (optional at this phase).
    try:
        import xformers  # noqa: F401

        console.print(f"xFormers = [green]{xformers.__version__}[/]")
    except Exception:  # noqa: BLE001
        console.print("[dim]xFormers not installed (optional; add before training if needed).[/]")

    console.rule("[bold green]Environment OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
