# -*- coding: utf-8 -*-
"""PySide6 desktop UI for the Anima LoRA trainer.

Two parent tabs — **Training** and **Utils** — over the shared, torch-free
:mod:`gui.backend`, so this panel emits the same ``train.py`` commands as the
Gradio one; only the UI differs (native dialogs, real tables, no localhost).

Training child tabs (curated fields + schema args routed in by keyword):
- **Folder**: every path/folder picker; sample / validation / save / logging /
  resume args land here.
- **Subset**: the subset table (→ ``form['subsets']``; per-subset multi-scale
  ``tiers`` + ``gradient_checkpointing``) + an Auto-preprocess toggle + global
  caption/shuffle flags.
- **Network**: adapter selection — method (LoRA type), network module/dim/alpha/
  args, LyCORIS preset + algo.
- **Optimizer**: the training-settings mega-tab — optimizer/scheduler (+args),
  loss/SNR/prior, the LR family + train-scope, norms/dropout, noise, and the
  core/hardware knobs (epochs/steps/batch/precision/swap/compile/seed) +
  flow-matching/timestep params.
- **Monitoring**: web-monitor flags.
- **Metadata**: metadata_* + no_metadata.
- **Extra**: everything uncaught (inference stacks: dcw/spectrum/spd/… ) + a raw
  ``extra_flags`` box.

Utils child tabs: Dataset (image+caption viewer/editor, tag sorter, mask overlay),
Preprocess (resize → VAE/TE/PE/pooled caches), Update (git pull + uv sync),
Auto-batch (multi-scale tier search + max-N blocks_to_swap / activation-budget
search), Masking (SAM3 + MIT). Right panel: command preview, Start/Stop, live log,
config TOML load/save, and a collapsible saved-run **Queue** (expands upward).

Schema args come from ``backend.list_arg_groups()`` (needs torch to populate);
without it the curated fields still render and the structure is intact.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QPropertyAnimation, Qt, QTimer, QUrl
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QFont,
    QKeySequence,
    QPalette,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui import backend
from gui.modules.arg_help import ARG_HELP  # Korean per-dest help (en fallback)
from gui.modules.config_io import load_toml_to_form, save_form_to_toml
from gui.native.tag_sort import KEEP_TOKENS_SEPARATOR


# --------------------------------------------------------------------------- #
# i18n — English / 한국어. tr(s) returns the Korean string when the language is
# "ko" and one is known, else the English text (so untranslated / deliberately
# technical strings — flag names, optimizer ids — stay English in both modes).
# The choice persists to gui/store/lang.txt and defaults to the OS locale.
# --------------------------------------------------------------------------- #
def _lang_file():
    return backend.STORE_DIR / "lang.txt"


def _load_lang() -> str:
    try:
        v = _lang_file().read_text(encoding="utf-8").strip().lower()
        if v in ("en", "ko"):
            return v
    except Exception:
        pass
    try:
        import locale

        if (locale.getdefaultlocale()[0] or "").lower().startswith("ko"):
            return "ko"
    except Exception:
        pass
    return "en"


def _save_lang(code: str) -> None:
    try:
        p = _lang_file()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(code, encoding="utf-8")
    except Exception:
        pass


_LANG = _load_lang()

_KO = {
    # window / nav
    "Anima LoRA — native trainer": "Anima LoRA — 네이티브 트레이너",
    "Training": "학습",
    "Utils": "도구",
    "Folder": "폴더",
    "Subset": "서브셋",
    "Network": "네트워크",
    "Optimizer": "옵티마이저",
    "Monitoring": "모니터링",
    "Metadata": "메타데이터",
    "Extra": "추가",
    "Experimental": "실험기능",
    # required-field validation
    "Required fields missing": "필수값 누락",
    "Fill these before starting:": "시작 전에 아래 값을 입력하세요:",
    "Sample every N steps/epochs": "샘플 N steps/epochs",
    "Validate every N steps/epochs": "검증 N steps/epochs",
    "Dataset": "데이터셋",
    "Preprocess": "전처리",
    "Update": "업데이트",
    "Auto-batch": "오토배치",
    "Masking": "마스킹",
    "Tools": "툴",
    # group titles
    "Model paths": "모델 경로",
    "Output / resume / logs": "출력 / 재개 / 로그",
    "Dataset / samples": "데이터셋 / 샘플",
    "Adapter": "어댑터",
    "Scheduler": "스케줄러",
    "Learning rates / scope": "학습률 / 범위",
    "Loss / regularization": "손실 / 정규화",
    "Core / hardware": "코어 / 하드웨어",
    "Web monitor": "웹 모니터",
    "More flags": "기타 플래그",
    "Find argument": "인자 찾기",
    "Find argument… (name / description)": "인자 찾기… (이름 / 설명)",
    # schema-arg cluster headers (group boxes under each tab)
    "misc": "기타",
    "Precision": "정밀도",
    "Batch & steps": "배치 & 스텝",
    "Memory · checkpointing · offload": "메모리 · 체크포인팅 · 오프로드",
    "Dataloader": "데이터로더",
    "VAE / TE encode & cache": "VAE / TE 인코딩 & 캐시",
    "Resume position": "재개 위치",
    "Learning rate & schedule": "학습률 & 스케줄",
    "Loss": "손실",
    "Timestep / flow-matching": "타임스텝 / flow-matching",
    "Noise": "노이즈",
    "Validation": "검증",
    "Sampling": "샘플링",
    "Save / checkpoints": "저장 / 체크포인트",
    "Logging": "로깅",
    "Data / paths": "데이터 / 경로",
    "Caption variants": "캡션 변형",
    "Captions": "캡션",
    "Per-layer LR": "레이어별 학습률",
    "Cond-diff loss": "Cond-diff 손실",
    "Functional loss": "Functional 손실",
    "LLM adapter": "LLM 어댑터",
    # field labels
    "DiT checkpoint": "DiT 체크포인트",
    "Text encoder (Qwen3)": "텍스트 인코더 (Qwen3)",
    "Tokenizer path": "토크나이저 경로",
    "Output name": "출력 이름",
    "Output dir": "출력 폴더",
    "Resume (state dir)": "재개 (상태 폴더)",
    "Warm-start weights": "웜스타트 가중치",
    "Dataset config TOML": "데이터셋 설정 TOML",
    "Sample prompts file": "샘플 프롬프트 파일",
    "Sample every N steps": "N 스텝마다 샘플",
    "Sample every N epochs": "N epoch마다 샘플",
    "Sample before training": "학습 전 샘플 생성",
    "Sample sampler": "샘플 샘플러",
    "Sample decode inline": "샘플 인라인 디코드",
    "LoRA type (method)": "LoRA 종류 (method)",
    "Network module": "네트워크 모듈",
    "Network dim (rank)": "네트워크 dim (rank)",
    "Network alpha": "네트워크 alpha",
    "LyCORIS preset": "LyCORIS 프리셋",
    "LyCORIS algo (loha/lokr/…)": "LyCORIS 알고리즘 (loha/lokr/…)",
    "↳ args help": "↳ 인자 도움말",
    "LR scheduler (builtin)": "LR 스케줄러 (내장)",
    "LR scheduler (custom dotted path)": "LR 스케줄러 (커스텀 경로)",
    "Warmup steps": "워밍업 스텝",
    "Learning rate": "학습률",
    "UNet / DiT LR": "UNet / DiT 학습률",
    "Text-encoder LR": "텍스트 인코더 학습률",
    "LLM-adapter LR": "LLM 어댑터 학습률",
    "Train scope": "학습 범위",
    "Loss type": "손실 종류",
    "Network dropout": "네트워크 드롭아웃",
    "Scale weight norms": "가중치 norm 스케일",
    "Max grad norm": "최대 grad norm",
    "Hardware preset": "하드웨어 프리셋",
    "Max epochs": "최대 epoch",
    "Max steps": "최대 step",
    "Batch size": "배치 크기",
    "Grad accumulation": "Grad 누적",
    "Blocks to swap": "스왑 블록 수",
    "Seed": "시드",
    "Mixed precision": "혼합 정밀도",
    "Enable (--monitor)": "활성화 (--monitor)",
    "Host": "호스트",
    "Port": "포트",
    # run panel
    "Language": "언어",
    "Load config…": "설정 불러오기…",
    "Save config…": "설정 저장…",
    "Command preview": "명령 미리보기",
    "Preview": "미리보기",
    "▶ Start": "▶ 시작",
    "■ Stop": "■ 중지",
    "Open monitor": "모니터 열기",
    "Log": "로그",
    "▲ Queue": "▲ 대기열",
    "idle": "대기 중",
    # subset cards
    "Subsets": "서브셋",
    "SUBSET": "서브셋",
    "➕ Add all subfolders from a folder…": "➕ 폴더의 모든 하위폴더 추가…",
    "➕ Add subset": "➕ 서브셋 추가",
    "Input image dir": "입력 이미지 폴더",
    "Cache dir": "캐시 폴더",
    "Number of repeats": "반복 횟수",
    "Keep tokens": "Keep tokens 수",
    "Caption extension": "캡션 확장자",
    "Caption dropout rate": "캡션 드롭아웃 비율",
    "Flip augment": "좌우 반전",
    "Random crop": "랜덤 크롭",
    "Grad checkpointing": "그래디언트 체크포인팅",
    "Validation set (hold out)": "검증셋으로 사용 (전체 분리)",
    "▸ Optional args": "▸ 선택 인자",
    "▾ Optional args": "▾ 선택 인자",
    "Tiers (multi-scale)": "타일 (멀티스케일)",
    # loss / timestep / weighting group + scheduler
    "LR scheduler": "LR 스케줄러",
    "Constant→cosine (one-shot)": "Constant→cosine (원샷)",
    "↳ cosine tail (epochs)": "↳ cosine tail (epoch)",
    "LR decay steps": "LR decay 스텝",
    "Min LR ratio": "최소 LR 비율",
    "Scheduler timescale": "스케줄러 timescale",
    "Caching / memory": "캐싱 / 메모리",
    "Cache VAE latents": "VAE latent 캐시",
    "Cache text-encoder outputs": "텍스트 인코더 출력 캐시",
    "Gradient checkpointing": "그래디언트 체크포인팅",
    "Qwen image VAE (2D)": "Qwen 이미지 VAE (2D)",
    "Qwen3 max token length": "Qwen3 최대 토큰 길이",
    "Attention": "어텐션",
    "Attention mode": "어텐션 방식",
    "↳ optimizer args help": "↳ 옵티마이저 인자 도움말",
    "↳ scheduler args help": "↳ 스케줄러 인자 도움말",
    "▸ Show usable args": "▸ 사용 가능한 인자 보기",
    "▾ Hide args": "▾ 인자 숨기기",
    "Pick an optimizer / scheduler first.": "옵티마이저 / 스케줄러를 먼저 선택하세요.",
    "➕ Add arg": "➕ 인자 추가",
    "Loss / timestep / weighting": "손실 / 타임스텝 / 가중치",
    "Huber c": "Huber c",
    "Huber schedule": "Huber 스케줄",
    "Timestep sampling": "타임스텝 샘플링",
    "Sigmoid scale": "시그모이드 스케일",
    "Weighting scheme": "가중치 스킴",
    "Logit mean": "Logit 평균",
    "Logit std": "Logit 표준편차",
    # greying reasons (shown inline on the disabled field's label)
    "needs loss_type = huber / smooth_l1": "loss_type이 huber / smooth_l1일 때만 사용됩니다",
    "needs timestep_sampling = sigmoid / shift / flux_shift": "timestep_sampling이 sigmoid / shift / flux_shift일 때만 사용됩니다",
    "needs timestep_sampling = shift": "timestep_sampling이 shift일 때만 사용됩니다",
    "needs weighting_scheme = logit_normal": "weighting_scheme이 logit_normal일 때만 사용됩니다",
    "needs weighting_scheme = mode": "weighting_scheme이 mode일 때만 사용됩니다",
    "needs ip_noise_gamma > 0": "ip_noise_gamma > 0일 때만 사용됩니다",
    "needs network_weights set": "network_weights를 지정해야 사용됩니다",
    "needs cosine_with_restarts / cosine_with_min_lr / WSD": "cosine_with_restarts / cosine_with_min_lr / warmup_stable_decay일 때만 사용됩니다",
    "needs lr_scheduler = polynomial": "lr_scheduler가 polynomial일 때만 사용됩니다",
    "disabled while use_constantcosine is on (it replaces the scheduler)": "use_constantcosine을 켜면 비활성화됩니다(스케줄러를 대체)",
}


def tr(s: str) -> str:
    """Translate a UI string to the active language (English fallback)."""
    return _KO.get(s, s) if _LANG == "ko" else s


# --------------------------------------------------------------------------- #
# Curated layout — (tab, [(group_title, [(dest, label, kind), …]), …]).
# kind ∈ text | combo:<src> | tristate | bool | file | dir | scope | opthelp.
# Schema args (list_arg_groups) are routed in on top via _ROUTE_RULES; any dest
# placed here is excluded from that routing (no double-render).
# --------------------------------------------------------------------------- #
_TRAINING_TABS: list[tuple[str, list[tuple[str, list[tuple[str, str, str]]]]]] = [
    (
        "Folder",
        [
            (
                "Model paths",
                [
                    ("dit_path", "DiT checkpoint", "file"),
                    ("te_path", "Text encoder (Qwen3)", "file"),
                    ("vae_path", "VAE", "file"),
                    ("t5_tokenizer_path", "Tokenizer path", "file"),
                ],
            ),
            (
                "Output / resume / logs",
                [
                    ("output_name", "Output name", "text"),
                    ("output_dir", "Output dir", "dir"),
                    ("resume", "Resume (state dir)", "dir"),
                    ("network_weights", "Warm-start weights", "file"),
                ],
            ),
            (
                "Dataset",
                [
                    ("dataset_config", "Dataset config TOML", "file"),
                ],
            ),
            (
                # All sample-IMAGE-generation knobs consolidated here (per user).
                # The cadence/sampler/at_first/decode flags are emitted from the form
                # by backend._method_preset_extra (they're in _CURATED_ARGS).
                "Sampling",
                [
                    ("sample_prompts", "Sample prompts file", "file"),
                    ("sample_every_n_steps", "Sample every N steps", "text"),
                    ("sample_every_n_epochs", "Sample every N epochs", "text"),
                    ("sample_at_first", "Sample before training", "bool"),
                    ("sample_sampler", "Sample sampler", "combo:euler,er_sde,lcm"),
                    (
                        "sample_decode_inline",
                        "Sample decode inline",
                        "combo:auto,true,false",
                    ),
                ],
            ),
            (
                # Optional feature — collapsed behind an Enable checkbox (per user).
                "Validation",
                [
                    ("validate_every_n_steps", "Validate every N steps", "text"),
                    ("validate_every_n_epochs", "Validate every N epochs", "text"),
                    ("validation_split", "Validation split", "text"),
                    ("validation_split_num", "Validation split num", "text"),
                    ("validation_seed", "Validation seed", "text"),
                    ("validation_cfg_scale", "Validation CFG scale", "text"),
                    ("validation_sample_steps", "Validation sample steps", "text"),
                    ("max_validation_steps", "Max validation steps", "text"),
                    ("use_cmmd", "Use CMMD metric", "bool"),
                ],
            ),
        ],
    ),
    (
        "Subset",
        [
            (
                "Preprocess",
                [
                    (
                        "auto_preprocess",
                        "Auto-preprocess on Start (resize → cache per subset tiers)",
                        "bool",
                    ),
                    (
                        "auto_keep_tokens",
                        "Auto keep_tokens (emit --keep_tokens_separator for the "
                        "Dataset-tab-inserted separator)",
                        "bool",
                    ),
                ],
            ),
            (
                # Image management = subset; caches are tied to the images, so the
                # VAE/TE caches + memory knobs live here (user's grouping).
                "Caching / memory",
                [
                    ("use_vae_cache", "Cache VAE latents", "tristate"),
                    ("use_text_cache", "Cache text-encoder outputs", "tristate"),
                    ("gradient_checkpointing", "Gradient checkpointing", "bool"),
                    ("qwen_image_vae_2d", "Qwen image VAE (2D)", "bool"),
                    ("qwen3_max_token_length", "Qwen3 max token length", "text"),
                ],
            ),
        ],
    ),
    (
        "Network",
        [
            (
                "Adapter",
                [
                    ("method", "LoRA type (method)", "combo:methods"),
                    ("network_module", "Network module", "combo:network_modules"),
                    ("network_dim", "Network dim (rank)", "text"),
                    ("network_alpha", "Network alpha", "text"),
                    ("network_args", "network_args", "kvblock"),
                ],
            ),
            (
                "LyCORIS",
                [
                    ("lycoris_preset", "LyCORIS preset", "combo:lycoris_presets"),
                    ("algo", "LyCORIS algo (loha/lokr/…)", "combo:lycoris_algos"),
                ],
            ),
        ],
    ),
    (
        "Optimizer",
        [
            (
                "Optimizer",
                [
                    ("optimizer_type", "Optimizer", "combo:optimizers"),
                    ("optimizer_args", "optimizer_args", "kvblock"),
                    (
                        "optimizer_args",
                        "↳ optimizer args help",
                        "opthelp:optimizer_type",
                    ),
                ],
            ),
            (
                "Scheduler",
                [
                    # One field: builtin names (cosine, …) AND custom dotted paths in the
                    # same list — the backend emits --lr_scheduler vs --lr_scheduler_type
                    # by the "." heuristic, so no separate "(builtin)" box is needed.
                    ("lr_scheduler_type", "LR scheduler", "combo:schedulers"),
                    # constant→cosine one-shot lives WITH the scheduler (it replaces it).
                    ("use_constantcosine", "Constant→cosine (one-shot)", "bool"),
                    ("constantcosine_tail_epochs", "↳ cosine tail (epochs)", "text"),
                    ("lr_scheduler_args", "lr_scheduler_args", "kvblock"),
                    (
                        "lr_scheduler_args",
                        "↳ scheduler args help",
                        "opthelp:lr_scheduler_type",
                    ),
                    ("lr_warmup_steps", "Warmup steps", "text"),
                    # scheduler-cluster flags pulled out of "More flags" to sit here.
                    ("lr_decay_steps", "LR decay steps", "text"),
                    ("lr_scheduler_min_lr_ratio", "Min LR ratio", "text"),
                    ("lr_scheduler_timescale", "Scheduler timescale", "text"),
                ],
            ),
            (
                "Learning rates / scope",
                [
                    ("learning_rate", "Learning rate", "text"),
                    ("unet_lr", "UNet / DiT LR", "text"),
                    ("text_encoder_lr", "Text-encoder LR", "text"),
                    ("llm_adapter_lr", "LLM-adapter LR", "text"),
                    ("__scope__", "Train scope", "scope"),
                ],
            ),
            (
                # Loss + the timestep/weighting knobs that belong WITH it (huber_c /
                # huber_schedule gate off loss_type; sigmoid_scale off timestep_sampling;
                # logit_mean/std off weighting_scheme) — grouped here instead of leaking
                # to the catch-all "More flags".
                "Loss / timestep / weighting",
                [
                    ("loss_type", "Loss type", "combo:l2,huber,smooth_l1"),
                    ("huber_c", "Huber c", "text"),
                    (
                        "huber_schedule",
                        "Huber schedule",
                        "combo:constant,exponential,snr",
                    ),
                    ("network_dropout", "Network dropout", "text"),
                    ("scale_weight_norms", "Scale weight norms", "text"),
                    ("max_grad_norm", "Max grad norm", "text"),
                    (
                        "timestep_sampling",
                        "Timestep sampling",
                        "combo:sigmoid,uniform,logit_normal,shift",
                    ),
                    ("sigmoid_scale", "Sigmoid scale", "text"),
                    (
                        "weighting_scheme",
                        "Weighting scheme",
                        "combo:logit_normal,mode,cosmap,sigma_sqrt,none",
                    ),
                    ("logit_mean", "Logit mean", "text"),
                    ("logit_std", "Logit std", "text"),
                    ("t_min", "t_min (σ)", "text"),
                    ("t_max", "t_max (σ)", "text"),
                ],
            ),
            (
                "Core / hardware",
                [
                    ("preset", "Hardware preset", "combo:presets"),
                    ("max_train_epochs", "Max epochs", "text"),
                    ("max_train_steps", "Max steps", "text"),
                    ("train_batch_size", "Batch size", "text"),
                    ("gradient_accumulation_steps", "Grad accumulation", "text"),
                    ("blocks_to_swap", "Blocks to swap", "text"),
                    ("seed", "Seed", "text"),
                    ("mixed_precision", "Mixed precision", "combo:bf16,fp16,no"),
                    ("torch_compile", "torch.compile", "tristate"),
                ],
            ),
        ],
    ),
    (
        "Monitoring",
        [
            (
                "Web monitor",
                [
                    ("monitor", "Enable (--monitor)", "bool"),
                    ("monitor_host", "Host", "text"),
                    ("monitor_port", "Port", "text"),
                ],
            ),
        ],
    ),
    (
        # consolidated anima-specific flags; attention knobs pinned at the top.
        "anima_lora",
        [
            (
                "Attention",
                [
                    (
                        "attn_mode",
                        "Attention mode",
                        "combo:torch,flash,sageattn,flex,sdpa",
                    )
                ],
            ),
        ],
    ),
    ("Experimental", []),
    ("Metadata", []),
    ("Extra", []),
]

# Schema-arg → tab routing (ordered; first include-match wins, exclude vetoes).
# Mirrors the user's spec: Folder = paths/sample/valid/save/log; Optimizer = the
# training mega-tab; Network ≈ curated only; Metadata/Monitoring narrow; rest →
# Extra. Adjust the keyword lists to re-group.
_ROUTE_RULES: list[tuple[str, list[str], list[str]]] = [
    (
        "Folder",
        [
            "_dir",
            "_path",
            "sample",
            "valid",
            "cmmd",
            "save",
            "output",
            "huggingface",
            "hub_",
            "resume",
            # save-resume cadence — a SAVE knob, NOT memory checkpointing; pin it to
            # Folder so the Optimizer rule's bare "checkpointing" can't steal it.
            "checkpointing_epochs",
            "logging",
            "log_tracker",
            "console_log",
            "log_with",
            "log_prefix",
            "log_every",
            "in_json",
            "wandb",
        ],
        ["logit", "sample_ratio"],
    ),
    ("Monitoring", ["monitor"], []),
    ("Metadata", ["metadata"], []),
    (
        "Subset",
        [
            "caption",
            "shuffle",
            "weighted_caption",
            "token_warmup",
            "secondary_separator",
            "keep_tokens_separator",
            "wildcard",
        ],
        [],
    ),
    (
        "Optimizer",
        [
            "optimizer",
            "scheduler",
            "lr_",
            "_lr",
            "loss",
            "huber",
            "snr",
            "prior",
            "noise",
            "warmup",
            "decay",
            "debiased",
            "grad_norm",
            "scale_weight",
            "dropout",
            "unet_only",
            "text_encoder_only",
            "train_text_encoder",
            "timestep",
            "sigmoid",
            "weighting",
            "logit",
            "t_min",
            "t_max",
            "discrete_flow",
            "mode_scale",
            "qwen3_max_token",
            "batch",
            "blocks_to_swap",
            "block_swap",
            "checkpointing",
            "compile",
            "dynamo",
            "cudagraph",
            "mixed_precision",
            "full_bf16",
            "full_fp16",
            "fp8",
            "seed",
            "dataloader",
            "pin_memory",
            "prefetch",
            "num_workers",
            "cache",
            "accumulation",
            "max_train",
            "initial_",
            "lowram",
            "highvram",
            "offload",
            "fused",
            "activation_memory",
            "persistent",
            "unsloth",
            "channel_scal",
            # speed / VRAM knobs → sit next to compile (per user)
            "no_half_vae",
            "attn_mode",
            "attn_softmax",
            "split_attn",
            "sdpa",
            "sageattn",
            "qwen_image_vae",
            "vae_chunk",
            "vae_disable_cache",
            "text_encoder_cpu",
            # constant→cosine one-shot → sits next to the scheduler (per user)
            "constantcosine",
        ],
        ["caption", "sample_decode"],
    ),
    ("Network", ["network", "lycoris", "conv_dim", "conv_alpha"], []),
]


def _route_tab(dest: str) -> str:
    for tab, inc, exc in _ROUTE_RULES:
        if any(k in dest for k in inc) and not any(k in dest for k in exc):
            return tab
    return "Extra"


# Groups rendered as a checkable/collapsible box: hidden + inert (excluded from the
# command) until the user ticks the title checkbox. For non-essential features
# (sampling, validation) so they don't clutter the always-visible essential knobs.
_OPTIONAL_GROUPS = {"Sampling", "Validation"}

# Experimental-feature flag families (the repo's `exp-*` set — BYG, Soft Tokens,
# SPD, DirectEdit/inversion). Their schema flags route to the Experimental tab even
# when they're Anima-specific (this check wins over the anima_lora-tab routing).
_EXPERIMENTAL_KEYS = ("byg", "soft_tokens", "spd", "directedit", "inversion")
_EXPERIMENTAL_TAB = "Experimental"


def _is_experimental(dest: str) -> bool:
    return any(k in dest for k in _EXPERIMENTAL_KEYS)


# Subset table columns → keys consumed by backend._dataset_subsets.
_SUBSET_COLS = [
    ("image_dir", "image_dir"),
    ("cache_dir", "cache_dir"),
    ("num_repeats", "num_repeats"),
    ("keep_tokens", "keep_tokens"),
    ("caption_extension", "caption_ext"),
    ("caption_dropout_rate", "cap_dropout"),
    ("batch_size", "batch_size"),
    ("tiers", "tiers (multi-scale, e.g. 512,1024)"),
    ("flip_aug", "flip_aug"),
    ("random_crop", "random_crop"),
    ("gradient_checkpointing", "grad_ckpt"),
]
_SUBSET_BOOL_COLS = {"flip_aug", "random_crop", "gradient_checkpointing"}
# Flags handled by the curated train-scope combo (kept out of schema routing).
_SCOPE_FLAGS = {"network_train_unet_only", "network_train_text_encoder_only"}


def _truthy(v: object) -> bool:
    return bool(v) and str(v).lower() not in ("false", "0", "")


def _pos(v: object) -> bool:
    """True when v parses to a number > 0 (a blank / 0 / non-numeric → False).
    Used for ``> 0`` gates where _truthy is wrong (e.g. the string "0.0")."""
    try:
        return float(str(v)) > 0
    except (TypeError, ValueError):
        return False


# Conflict / dependency greying — ported from the Gradio panel's _interactive_states.
# (target dest, predicate(driver values) → enabled). A greyed target is disabled AND
# excluded from the launch command (its value defers to the config chain).
# (target dest, predicate(driver values) → enabled, reason-when-disabled). The reason
# is tr()'d and shown as the disabled field's tooltip so the user sees WHY it's greyed.
_GREY_RULES: list[tuple[str, object, str]] = [
    (
        "huber_c",
        lambda v: v.get("loss_type") in ("huber", "smooth_l1"),
        "needs loss_type = huber / smooth_l1",
    ),
    (
        "huber_schedule",
        lambda v: v.get("loss_type") in ("huber", "smooth_l1"),
        "needs loss_type = huber / smooth_l1",
    ),
    (
        "huber_scale",
        lambda v: v.get("loss_type") in ("huber", "smooth_l1"),
        "needs loss_type = huber / smooth_l1",
    ),
    # σ sampling: sigmoid_scale / sigmoid_bias feed the sigmoid|shift|flux_shift
    # branches (noise.py get_noisy_model_input_and_timesteps); "" = default sigmoid.
    (
        "sigmoid_scale",
        lambda v: v.get("timestep_sampling") in ("", "sigmoid", "shift", "flux_shift"),
        "needs timestep_sampling = sigmoid / shift / flux_shift",
    ),
    (
        "sigmoid_bias",
        lambda v: v.get("timestep_sampling") in ("", "sigmoid", "shift", "flux_shift"),
        "needs timestep_sampling = sigmoid / shift / flux_shift",
    ),
    # discrete_flow_shift is read ONLY by the `shift` branch (flux_shift uses a
    # resolution-derived mu instead — noise.py L111-122).
    (
        "discrete_flow_shift",
        lambda v: v.get("timestep_sampling") == "shift",
        "needs timestep_sampling = shift",
    ),
    # logit_mean/std + mode_scale feed the SD3 density path (the "sigma"/weighting
    # branch); each is live only under its own weighting_scheme.
    (
        "logit_mean",
        lambda v: v.get("weighting_scheme") == "logit_normal",
        "needs weighting_scheme = logit_normal",
    ),
    (
        "logit_std",
        lambda v: v.get("weighting_scheme") == "logit_normal",
        "needs weighting_scheme = logit_normal",
    ),
    (
        "mode_scale",
        lambda v: v.get("weighting_scheme") == "mode",
        "needs weighting_scheme = mode",
    ),
    # ip_noise_gamma_random_strength only randomises a gamma that must be > 0
    # (noise.py reads it solely inside `if args.ip_noise_gamma:` L159-166).
    (
        "ip_noise_gamma_random_strength",
        lambda v: _pos(v.get("ip_noise_gamma")),
        "needs ip_noise_gamma > 0",
    ),
    # dim_from_weights infers rank FROM an existing LoRA → needs network_weights.
    (
        "dim_from_weights",
        lambda v: bool(str(v.get("network_weights") or "").strip()),
        "needs network_weights set",
    ),
    # Built-in LR-scheduler shape params: num_cycles only feeds the restart/min-lr/
    # WSD families, power only feeds polynomial (schedulers.py L210-260). The combo
    # holds the builtin name (dotted = custom module → these are inert anyway).
    (
        "lr_scheduler_num_cycles",
        lambda v: (
            v.get("lr_scheduler_type")
            in ("cosine_with_restarts", "cosine_with_min_lr", "warmup_stable_decay")
        ),
        "needs cosine_with_restarts / cosine_with_min_lr / WSD",
    ),
    (
        "lr_scheduler_power",
        lambda v: v.get("lr_scheduler_type") == "polynomial",
        "needs lr_scheduler = polynomial",
    ),
    (
        "lr_scheduler_type",
        lambda v: (
            not _truthy(v.get("use_constantcosine"))
            and "schedulefree" not in str(v.get("optimizer_type") or "").lower()
        ),
        "off (constant→cosine / schedule-free optimizer)",
    ),
    # schedule-free optimizers run their own LR schedule → warmup is ignored.
    (
        "lr_warmup_steps",
        lambda v: "schedulefree" not in str(v.get("optimizer_type") or "").lower(),
        "ignored by schedule-free optimizer",
    ),
    # Auto-preprocess builds + reads the VAE/TE caches (and rewrites dataset_config),
    # so it manages these toggles — locked while it's on.
    (
        "use_vae_cache",
        lambda v: not _truthy(v.get("auto_preprocess")),
        "managed by Auto-preprocess",
    ),
    (
        "use_text_cache",
        lambda v: not _truthy(v.get("auto_preprocess")),
        "managed by Auto-preprocess",
    ),
    # Train scope (combo 0=both, 1=UNet only, 2=TE only): the other side's LR is unused.
    (
        "unet_lr",
        lambda v: v.get("__scope__") != 2,
        "TE-only scope: UNet LR unused",
    ),
    (
        "text_encoder_lr",
        lambda v: v.get("__scope__") != 1,
        "UNet-only scope: TE LR unused",
    ),
]
# Driver dests whose change re-evaluates the rules above + the subset-column greying.
# (__scope__ is the curated train-scope combo, injected by _apply_greying.)
_GREY_DRIVERS = [
    "loss_type",
    "timestep_sampling",
    "weighting_scheme",
    "use_constantcosine",
    "use_vae_cache",
    "use_text_cache",
    "auto_preprocess",
    "optimizer_type",
    "ip_noise_gamma",
    "network_weights",
    "lr_scheduler_type",
]
# Subset table columns greyed by a cache driver (live-encoding-only knobs are inert
# once the cache is on): (col_key, driver_dest).
_SUBSET_GREY = [
    ("random_crop", "use_vae_cache"),
    ("caption_dropout_rate", "use_text_cache"),
]

# Only anima_lora's OWN flags (from its DiT/Anima arg-adders, detected in the backend
# as options()["anima_dests"]) go to the dedicated "anima_lora" tab; the inherited base
# args keep their normal per-tab "More flags" placement.
_ANIMA_TAB = "anima_lora"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(tr("Anima LoRA — native trainer"))
        self.resize(1280, 880)
        self._options = backend.options()
        self._getters: dict[str, object] = {}
        self._setters: dict[str, object] = {}
        self._adv: list[tuple[dict, object]] = []
        self._scope: QComboBox | None = None
        self._widgets: dict[str, QWidget] = {}  # dest → editable widget (for greying)
        self._watch: dict[str, QWidget] = {}  # watch-party fields (NOT saved to config)
        # title → (checkable QGroupBox, [dests]) for collapsible optional features.
        self._opt_groups: dict[str, tuple] = {}
        # dest → (QLabel, base_text) for inline grey-reason display.
        self._field_labels: dict[str, tuple] = {}
        self._highlighted: list[str] = []  # dests flagged by required-field validation
        # Dests placed explicitly → excluded from schema routing (no double render).
        self._curated: set[str] = {"extra_flags", *_SCOPE_FLAGS}
        for _tab, groups in _TRAINING_TABS:
            for _title, fields in groups:
                for dest, _label, kind in fields:
                    if not (kind.startswith("opthelp") or kind == "scope"):
                        self._curated.add(dest)
        # Partition schema args (arg_groups) into Training tabs by route.
        self._tab_schema: dict[str, list[dict]] = {}
        anima_dests = set(self._options.get("anima_dests") or [])
        for group in self._options.get("arg_groups") or []:
            for arg in group.get("args") or []:
                d = arg.get("dest") or ""
                if d in self._curated:
                    continue
                # Experimental flag families (exp-* set) win first — even when they're
                # Anima-specific. Then ONLY anima_lora's own flags go to the anima_lora
                # tab; inherited sd-scripts/kohya base args stay on their routed tab.
                if _is_experimental(d):
                    tab = _EXPERIMENTAL_TAB
                elif d in anima_dests:
                    tab = _ANIMA_TAB
                else:
                    tab = _route_tab(d)
                self._tab_schema.setdefault(tab, []).append(arg)

        self._build_central()

        # Ctrl+F → find any argument by name/label/description and jump to it.
        self._find_sc = QShortcut(QKeySequence.Find, self)
        self._find_sc.activated.connect(self._show_search)
        self._search_dlg: QDialog | None = None

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._poll)
        self._timer.start()
        self._poll()

    # ----- parent tabs ---------------------------------------------------- #
    def _build_central(self) -> None:
        """(Re)build the tabs + run panel into the central widget. Re-callable so a
        language switch can rebuild the whole UI in the new language."""
        self._opthelp_panels: list[tuple[str, QWidget, QWidget]] = []
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_parent_tabs())
        splitter.addWidget(self._build_run_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)
        self._wire_greying()
        self._apply_greying()

    def _set_language(self, code: str) -> None:
        """Switch UI language live: persist + rebuild the central widget, preserving
        the current form values (scalar fields via the getter/setter registries)."""
        global _LANG
        if code not in ("en", "ko") or code == _LANG:
            return
        _LANG = code
        _save_lang(code)
        self.setWindowTitle(tr("Anima LoRA — native trainer"))
        try:
            saved = {d: g() for d, g in self._getters.items()}
        except Exception:
            saved = {}
        # Schema ("advanced") fields live in self._adv + self._widgets, NOT in the
        # getter/setter registries — so they used to RESET on a language switch (the
        # "budget + dropdowns get cleared" bug). Snapshot their values by dest and
        # restore them onto the rebuilt widgets below.
        saved_adv: dict[str, object] = {}
        try:
            for arg, _g in self._adv:
                d = arg.get("dest")
                if d:
                    saved_adv[d] = self._widget_value(d)
        except Exception:
            pass
        # Auto-batch panel widgets live in self._ab (its own dict, no getters) and the
        # tier toggles in self._ab_res_checks — both rebuilt by _build_autobatch_tab, so
        # they reset on a language switch too (the "budget + dropdowns cleared" bug).
        saved_ab: dict[str, object] = {}
        try:
            for k, wdg in getattr(self, "_ab", {}).items():
                if isinstance(wdg, QCheckBox):
                    saved_ab[k] = wdg.isChecked()
                elif isinstance(wdg, QComboBox):
                    saved_ab[k] = wdg.currentText()
                else:
                    saved_ab[k] = wdg.text()
        except Exception:
            pass
        saved_ab_res = [
            t for t, cb in getattr(self, "_ab_res_checks", []) if cb.isChecked()
        ]
        try:
            saved_subsets = self._collect_subsets()
        except Exception:
            saved_subsets = []
        view = self._capture_view_state()  # current tab + scroll positions
        if getattr(self, "_timer", None) is not None:
            self._timer.stop()
        # Reset the per-build registries (repopulated by _build_central → builders).
        self._getters = {}
        self._setters = {}
        self._adv = []
        self._scope = None
        self._widgets = {}
        self._watch = {}
        self._build_central()
        for d, v in saved.items():
            setter = self._setters.get(d)
            if setter is not None:
                try:
                    setter(v)
                except Exception:
                    pass
        for d, v in saved_adv.items():  # schema fields: set by dest (no setter exists)
            try:
                self._set_widget_value(d, v)
            except Exception:
                pass
        for k, v in saved_ab.items():  # auto-batch panel widgets (own dict)
            wdg = getattr(self, "_ab", {}).get(k)
            try:
                if isinstance(wdg, QCheckBox):
                    wdg.setChecked(bool(v))
                elif isinstance(wdg, QComboBox):
                    self._set_combo(wdg, v)
                elif wdg is not None:
                    wdg.setText(str(v or ""))
            except Exception:
                pass
        if saved_ab_res:
            for t, cb in getattr(self, "_ab_res_checks", []):
                cb.setChecked(t in saved_ab_res)
        for s in saved_subsets:  # subset cards aren't in the getter registry
            try:
                self._add_subset_card(s)
            except Exception:
                pass
        self._restore_view_state(view)  # restore the tab + scroll the user was on
        if getattr(self, "_timer", None) is not None:
            self._timer.start()

    # ----- view-state preservation across a language-switch rebuild ------- #
    def _capture_view_state(self) -> dict:
        st: dict = {}
        pt = getattr(self, "_parent_tabs", None)
        if pt is not None:
            st["parent"] = pt.currentIndex()
        ti = getattr(self, "_training_inner", None)
        if ti is not None:
            st["training"] = ti.currentIndex()
        ui = getattr(self, "_utils_inner", None)
        if ui is not None:
            st["utils"] = ui.currentIndex()
        st["scrolls"] = [
            sc.verticalScrollBar().value()
            for sc in getattr(self, "_training_scrolls", [])
        ]
        return st

    def _restore_view_state(self, st: dict) -> None:
        pt = getattr(self, "_parent_tabs", None)
        if pt is not None and "parent" in st:
            pt.setCurrentIndex(st["parent"])
        ti = getattr(self, "_training_inner", None)
        if ti is not None and "training" in st:
            ti.setCurrentIndex(st["training"])
        ui = getattr(self, "_utils_inner", None)
        if ui is not None and "utils" in st:
            ui.setCurrentIndex(st["utils"])
        scrolls = getattr(self, "_training_scrolls", [])
        vals = st.get("scrolls") or []
        # Scrollbar maximum isn't valid until the layout settles → defer the set.
        for sc, val in zip(scrolls, vals):
            QTimer.singleShot(
                0, lambda sc=sc, val=val: sc.verticalScrollBar().setValue(val)
            )

    def _set_widget_value(self, dest: str, value: object) -> None:
        """Set a field's value by dest onto whatever widget now backs it — used to
        restore schema/advanced fields across the language-switch rebuild, since those
        register no setter (only a getter in self._adv)."""
        w = self._widgets.get(dest)
        if isinstance(w, QCheckBox):
            w.setChecked(_truthy(value))
        elif isinstance(w, QComboBox):
            # A negatable schema arg renders as a default/on/off combo; a config bool
            # maps to on/off. Choices/text combos take the value verbatim.
            if isinstance(value, bool):
                self._set_combo(w, "on" if value else "off")
            else:
                self._set_combo(w, value)
        elif isinstance(w, QLineEdit):
            if isinstance(value, (list, tuple)):  # nargs arg → space-joined
                w.setText(" ".join(str(x) for x in value))
            else:
                w.setText(str(value if value is not None else ""))

    # ----- Ctrl+F argument search ----------------------------------------- #
    def _build_search_index(self) -> list[dict]:
        """Searchable field index: every curated + schema field with a live widget →
        its containing Training-tab index + searchable label + Korean help. Rebuilt on
        each open so it tracks the current language and any rebuild."""
        idx: list[dict] = []
        tab_order = {name: i for i, (name, _g) in enumerate(_TRAINING_TABS)}
        seen: set[str] = set()
        for tname, groups in _TRAINING_TABS:  # curated fields (carry a human label)
            for _title, fields in groups:
                for dest, label, _kind in fields:
                    w = self._widgets.get(dest)
                    if w is None or dest in seen:
                        continue
                    seen.add(dest)
                    idx.append(
                        {
                            "dest": dest,
                            "label": label,
                            "ko": ARG_HELP.get(dest, ""),
                            "tab": tab_order.get(tname, 0),
                            "w": w,
                        }
                    )
        for tname, args in (self._tab_schema or {}).items():  # schema (auto) fields
            ti = tab_order.get(tname)
            if ti is None:
                continue
            for arg in args:
                dest = arg.get("dest") or ""
                w = self._widgets.get(dest)
                if not dest or w is None or dest in seen:
                    continue
                seen.add(dest)
                idx.append(
                    {
                        "dest": dest,
                        "label": dest,  # arg name stays English
                        "ko": ARG_HELP.get(dest, (arg.get("help") or "")),
                        "tab": ti,
                        "w": w,
                    }
                )
        return idx

    def _show_search(self) -> None:
        if self._search_dlg is None:
            dlg = QDialog(self)
            dlg.setWindowTitle(tr("Find argument"))
            dlg.setModal(False)
            dlg.resize(480, 400)
            v = QVBoxLayout(dlg)
            edit = QLineEdit()
            edit.setPlaceholderText(tr("Find argument… (name / description)"))
            results = QListWidget()
            v.addWidget(edit)
            v.addWidget(results, 1)
            edit.textChanged.connect(self._search_filter)
            edit.returnPressed.connect(
                lambda: self._goto_search(results.item(0)) if results.count() else None
            )
            results.itemActivated.connect(self._goto_search)
            self._search_dlg = dlg
            self._search_edit = edit
            self._search_results = results
        self._search_data = self._build_search_index()
        self._search_filter(self._search_edit.text())
        self._search_dlg.show()
        self._search_dlg.raise_()
        self._search_edit.setFocus()
        self._search_edit.selectAll()

    def _search_filter(self, text: str) -> None:
        q = (text or "").strip().lower()
        self._search_results.clear()
        tab_names = [tr(n) for n, _g in _TRAINING_TABS]
        for entry in getattr(self, "_search_data", []):
            hay = f"{entry['dest']} {entry['label']} {entry['ko']}".lower()
            if q and q not in hay:
                continue
            tn = tab_names[entry["tab"]] if 0 <= entry["tab"] < len(tab_names) else ""
            disp = f"[{tn}] {entry['dest']}"
            if entry["label"] and entry["label"] != entry["dest"]:
                disp += f" — {tr(entry['label'])}"
            it = QListWidgetItem(disp)
            it.setData(Qt.UserRole, entry)
            self._search_results.addItem(it)
            if self._search_results.count() >= 80:
                break

    def _goto_search(self, item) -> None:
        if item is None:
            return
        entry = item.data(Qt.UserRole)
        if not entry:
            return
        if getattr(self, "_parent_tabs", None) is not None:
            self._parent_tabs.setCurrentIndex(0)  # Training parent
        ti = entry["tab"]
        if getattr(self, "_training_inner", None) is not None:
            self._training_inner.setCurrentIndex(ti)
        scrolls = getattr(self, "_training_scrolls", [])
        w = entry["w"]
        if 0 <= ti < len(scrolls):
            scrolls[ti].ensureWidgetVisible(w)
        # Flash a gold border so the field is easy to spot, then clear it.
        w.setStyleSheet("border: 2px solid #FACC15; border-radius: 4px;")
        QTimer.singleShot(1600, lambda w=w: w.setStyleSheet(""))
        try:
            w.setFocus()
        except Exception:
            pass
        if self._search_dlg is not None:
            self._search_dlg.hide()

    def _build_parent_tabs(self) -> QTabWidget:
        parent = QTabWidget()
        self._parent_tabs = parent  # kept so a language-switch can restore the tab
        parent.addTab(self._build_training_parent(), tr("Training"))
        parent.addTab(self._build_utils_parent(), tr("Utils"))
        return parent

    def _build_training_parent(self) -> QTabWidget:
        inner = QTabWidget()
        self._training_inner = inner
        self._training_scrolls = []  # per-tab QScrollArea → restore scroll on rebuild
        for tab_name, groups in _TRAINING_TABS:
            sc = self._scroll(self._build_training_tab(tab_name, groups))
            self._training_scrolls.append(sc)
            inner.addTab(sc, tr(tab_name))
        return inner

    # ----- saved-run queue (collapsible panel, not a tab) ----------------- #
    def _build_queue_panel(self) -> QWidget:
        w = QWidget()
        vbox = QVBoxLayout(w)
        vbox.setContentsMargins(0, 0, 0, 0)
        self._queue_list = QListWidget()
        self._queue_list.setMaximumHeight(140)
        vbox.addWidget(self._queue_list, 1)
        row1 = QHBoxLayout()
        b_add = QPushButton("➕ Add current")
        b_load = QPushButton("Load selected → form")
        b_rm = QPushButton("➖ Remove selected")
        b_add.clicked.connect(self._queue_add)
        b_load.clicked.connect(self._queue_load)
        b_rm.clicked.connect(self._queue_remove)
        for b in (b_add, b_load, b_rm):
            row1.addWidget(b)
        vbox.addLayout(row1)
        row2 = QHBoxLayout()
        b_refresh = QPushButton("Refresh")
        b_clear = QPushButton("Clear all")
        b_run = QPushButton("▶ Run queue")
        b_refresh.clicked.connect(self._queue_refresh)
        b_clear.clicked.connect(self._queue_clear)
        b_run.clicked.connect(self._queue_run)
        for b in (b_refresh, b_clear, b_run):
            row2.addWidget(b)
        vbox.addLayout(row2)
        self._queue_refresh()
        return w

    def _queue_refresh(self) -> None:
        self._queue_list.clear()
        for it in backend.queue_list():
            li = QListWidgetItem(f"#{it.get('id')}  {it.get('name')}")
            li.setData(Qt.UserRole, it)
            self._queue_list.addItem(li)

    def _queue_selected(self) -> dict | None:
        li = self._queue_list.currentItem()
        return li.data(Qt.UserRole) if li else None

    def _queue_add(self) -> None:
        name, ok = QInputDialog.getText(self, "Queue", "Job name:")
        if not ok:
            return
        backend.queue_add(name.strip(), self._collect())
        self._queue_refresh()

    def _queue_load(self) -> None:
        it = self._queue_selected()
        if it and isinstance(it.get("form"), dict):
            self._apply(it["form"])
            self._do_preview()

    def _queue_remove(self) -> None:
        it = self._queue_selected()
        if it:
            backend.queue_remove(it.get("id"))
            self._queue_refresh()

    def _queue_clear(self) -> None:
        backend.queue_clear()
        self._queue_refresh()

    def _queue_run(self) -> None:
        res = backend.queue_run()
        if not res.get("ok"):
            QMessageBox.warning(self, "Queue", str(res.get("error") or res))

    def _build_training_tab(self, tab_name: str, groups: list) -> QWidget:
        w = QWidget()
        vbox = QVBoxLayout(w)
        for title, fields in groups:
            vbox.addWidget(self._build_group(title, fields))
        if tab_name == "Subset":
            vbox.addWidget(self._build_subset_box())
        # Schema args routed into this tab — grouped into a box PER CLUSTER so related
        # flags sit together (label | short value field | help text beside it).
        schema = self._tab_schema.get(tab_name) or []
        if schema:
            by_cluster: dict[str, list[dict]] = {}
            for arg in schema:
                by_cluster.setdefault(arg.get("cluster") or "misc", []).append(arg)
            for cluster in sorted(by_cluster):
                box = QGroupBox(tr(str(cluster)))
                grid = QGridLayout(box)
                grid.setHorizontalSpacing(14)
                grid.setVerticalSpacing(8)
                grid.setColumnStretch(2, 1)  # description column absorbs the width
                rows = sorted(by_cluster[cluster], key=lambda a: a.get("dest") or "")
                for r, arg in enumerate(rows):
                    lbl = QLabel(
                        arg.get("dest") or arg.get("flag")
                    )  # arg name: English
                    fw = self._build_adv_field(arg)
                    fw.setMaximumWidth(200)
                    # Description in Korean when available (ARG_HELP), else the English
                    # argparse help. The arg NAME stays English either way.
                    htext = (arg.get("help") or "").strip()
                    if _LANG == "ko":
                        htext = ARG_HELP.get(arg.get("dest") or "", htext)
                    desc = QLabel(htext)
                    desc.setObjectName("argDesc")
                    desc.setWordWrap(True)
                    grid.addWidget(lbl, r, 0)
                    grid.addWidget(fw, r, 1)
                    grid.addWidget(desc, r, 2)
                vbox.addWidget(box)
        if tab_name == "Extra":
            vbox.addWidget(self._build_extra_flags_box())
        if tab_name == "Monitoring":
            vbox.addWidget(self._build_watch_party_box())
        vbox.addStretch(1)
        return w

    # ----- AI watch party (Claude + GPT) ---------------------------------- #
    def _build_watch_party_box(self) -> QGroupBox:
        gb = QGroupBox("AI watch party (Claude + GPT) — needs --monitor running")
        form = QFormLayout(gb)
        ak = QLineEdit()
        ak.setEchoMode(QLineEdit.Password)
        ak.setPlaceholderText("ANTHROPIC_API_KEY (not saved to config)")
        ok = QLineEdit()
        ok.setEchoMode(QLineEdit.Password)
        ok.setPlaceholderText("OPENAI_API_KEY (not saved to config)")
        self._watch["ANTHROPIC_API_KEY"] = ak
        self._watch["OPENAI_API_KEY"] = ok
        form.addRow("Anthropic key", ak)
        form.addRow("OpenAI key", ok)
        for label, key, default in (
            ("Interval (s)", "watch_interval", "30"),
            ("Turns per round", "watch_turns", "1"),
            ("Max rounds (0=∞)", "watch_rounds", "0"),
        ):
            e = QLineEdit(default)
            self._watch[key] = e
            form.addRow(label, e)
        # Default ON (privacy): sample images otherwise leave the machine to
        # Anthropic + OpenAI. The user opts in to sending them by unchecking this.
        no_img = QCheckBox(
            "Don't send sample images (privacy — they go to Anthropic+OpenAI)"
        )
        no_img.setChecked(True)
        self._watch["watch_no_images"] = no_img
        form.addRow(no_img)
        btn = QPushButton("▶ Start watch party")
        btn.clicked.connect(self._do_watch_party)
        form.addRow(btn)
        return gb

    def _do_watch_party(self) -> None:
        form = {}
        for key, w in self._watch.items():
            if isinstance(w, QCheckBox):
                form[key] = w.isChecked()
            else:
                form[key] = w.text().strip()
        res = backend.run_watch_party(form)
        if not res.get("ok"):
            QMessageBox.warning(self, "Watch party", str(res.get("error") or res))

    def _scroll(self, inner: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        return scroll

    # ----- curated field widgets ------------------------------------------ #
    def _build_group(self, title: str, fields: list[tuple[str, str, str]]) -> QGroupBox:
        gb = QGroupBox(tr(title))
        optional = title in _OPTIONAL_GROUPS
        if optional:
            # Checkable title → collapse: unchecked hides the body AND disables its
            # widgets (Qt), so the fields are excluded from the command (inert).
            gb.setCheckable(True)
            gb.setChecked(False)
            outer = QVBoxLayout(gb)
            outer.setContentsMargins(0, 0, 0, 0)
            body = QWidget()
            outer.addWidget(body)
            grid = QGridLayout(body)
        else:
            grid = QGridLayout(gb)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)
        # Trailing spacer column absorbs slack so the capped value fields stay tight
        # and the label/field pairs sit left-packed (no over-wide fields). Labels take
        # their natural width, so a long label just makes that pair a bit wider — the
        # proportions adapt instead of forcing a rigid 50/50.
        grid.setColumnStretch(4, 1)
        r = 0
        c = 0  # 0 = left pair (cols 0-1), 2 = right pair (cols 2-3)
        for dest, label, kind in fields:
            w = self._build_field(dest, kind)
            lbl = QLabel(tr(label))
            # Register the label so greying can show its reason inline (and so the
            # base text can be restored when re-enabled). First occurrence wins.
            self._field_labels.setdefault(dest, (lbl, tr(label)))
            # Path pickers / args-help / scope span the full width; the compact fields
            # (text/combo/bool/tristate) — no per-field description — pack TWO per row.
            if kind in ("file", "dir", "scope", "kvblock") or kind.startswith(
                "opthelp"
            ):
                if c != 0:
                    r += 1
                    c = 0
                grid.addWidget(lbl, r, 0)
                grid.addWidget(w, r, 1, 1, 4)
                r += 1
            else:
                grid.addWidget(lbl, r, c)
                grid.addWidget(w, r, c + 1)
                if c == 0:
                    c = 2
                else:
                    c = 0
                    r += 1
        if optional:
            gb.toggled.connect(body.setVisible)
            body.setVisible(False)
            self._opt_groups[title] = (gb, [d for d, _l, _k in fields])
        return gb

    def _set_combo(self, combo: QComboBox, value) -> None:
        """Select a value on a non-editable combo, adding it as an item first if it's
        not in the predefined list — so loading a config with a custom value (e.g. a
        dotted-path optimizer not in the zoo list) preserves it instead of losing it."""
        v = str(value or "")
        if v and combo.findText(v) < 0:
            combo.addItem(v)
        combo.setCurrentText(v)

    def _build_field(self, dest: str, kind: str) -> QWidget:
        if kind == "scope":
            combo = _Combo()
            combo.addItems(["both (UNet + TE)", "UNet only", "TE only"])
            self._scope = combo
            return combo
        if kind.startswith("opthelp"):
            # Inline, expands DOWNWARD (not a popup): lists the accepted args of the
            # selected optimizer OR scheduler. Source dest after the colon:
            # "opthelp:optimizer_type" or "opthelp:lr_scheduler_type" (default optimizer).
            source = kind.split(":", 1)[1] if ":" in kind else "optimizer_type"
            box = QWidget()
            v = QVBoxLayout(box)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(4)
            btn = QPushButton(tr("▸ Show usable args"))
            btn.setObjectName("subOpt")
            btn.setCheckable(True)
            body = QLabel()
            body.setObjectName("optHelpBody")
            body.setWordWrap(True)
            body.setVisible(False)
            body.setTextInteractionFlags(Qt.TextSelectableByMouse)

            def _toggle(on, b=btn, lb=body, src=source):
                if on:
                    lb.setText(self._arg_help_text(src))
                    b.setText(tr("▾ Hide args"))
                else:
                    b.setText(tr("▸ Show usable args"))
                lb.setVisible(on)

            btn.toggled.connect(_toggle)
            v.addWidget(btn)
            v.addWidget(body)
            # Register so changing the source optimizer/scheduler auto-refreshes the
            # open panel (no collapse + re-expand needed).
            self._opthelp_panels.append((source, btn, body))
            return box
        if kind == "kvblock":
            return self._build_kv_block(dest)
        if kind == "bool":
            cb = QCheckBox()
            self._getters[dest] = lambda c=cb: c.isChecked()
            self._setters[dest] = lambda v, c=cb: c.setChecked(_truthy(v))
            self._widgets[dest] = cb
            return cb
        if kind == "tristate":
            combo = _Combo()
            combo.addItems(["", "on", "off"])
            combo.setMaximumWidth(150)
            self._getters[dest] = lambda c=combo: c.currentText().strip()
            self._setters[dest] = lambda v, c=combo: c.setCurrentText(str(v or ""))
            self._widgets[dest] = combo
            return combo
        if kind.startswith("combo:"):
            src = kind.split(":", 1)[1]
            items = self._options.get(src) if src in self._options else src.split(",")
            combo = _Combo()  # non-editable (use_cmmd style)
            combo.setMaximumWidth(360)  # don't stretch to the longest dotted-path item
            combo.addItem("")
            combo.addItems([str(x) for x in (items or [])])
            self._getters[dest] = lambda c=combo: c.currentText().strip()
            self._setters[dest] = lambda v, c=combo: self._set_combo(c, v)
            self._widgets[dest] = combo
            return combo
        edit = QLineEdit()
        self._getters[dest] = lambda e=edit: e.text().strip()
        self._setters[dest] = lambda v, e=edit: e.setText(str(v or ""))
        self._widgets[dest] = edit
        if kind in ("file", "dir"):
            row = QWidget()
            hb = QHBoxLayout(row)
            hb.setContentsMargins(0, 0, 0, 0)
            hb.addWidget(edit)
            btn = QPushButton("📁")
            btn.setObjectName("icon")
            btn.setFixedWidth(40)
            btn.clicked.connect(lambda _=False, e=edit, k=kind: self._browse(e, k))
            hb.addWidget(btn)
            return row
        # Plain value field — keep it compact (not stretched across the panel). Path
        # fields are the file/dir rows above, which stay full-width.
        edit.setMinimumWidth(200)
        edit.setMaximumWidth(360)
        return edit

    def _build_kv_block(self, dest: str) -> QWidget:
        """A key=value block editor (like the kohya_ss 'NETWORK ARGS' panel): an Add
        button + an unlimited list of key/value rows, each deletable. Collected as
        newline-joined ``key=value`` tokens (what the backend's _arg_split consumes)."""
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        add = QPushButton(tr("➕ Add arg"))
        add.setObjectName("subOpt")
        holder = QWidget()
        rows_lay = QVBoxLayout(holder)
        rows_lay.setContentsMargins(0, 0, 0, 0)
        rows_lay.setSpacing(4)
        rows: list[dict] = []

        def _add_row(key: str = "", val: str = "") -> None:
            rw = QWidget()
            h = QHBoxLayout(rw)
            h.setContentsMargins(0, 0, 0, 0)
            ke = QLineEdit(key)
            ke.setPlaceholderText("key")
            ve = QLineEdit(val)
            ve.setPlaceholderText("value")
            dele = QPushButton("🗑")
            dele.setObjectName("icon")
            dele.setFixedWidth(40)
            h.addWidget(ke, 1)
            h.addWidget(ve, 1)
            h.addWidget(dele)
            entry = {"w": rw, "k": ke, "v": ve}
            dele.clicked.connect(
                lambda _=False, e=entry: (
                    rows.remove(e) if e in rows else None,
                    e["w"].setParent(None),
                    e["w"].deleteLater(),
                )
            )
            rows.append(entry)
            rows_lay.addWidget(rw)

        add.clicked.connect(lambda *_: _add_row())
        v.addWidget(add)
        v.addWidget(holder)

        def _get() -> str:
            out = []
            for e in rows:
                k = e["k"].text().strip()
                val = e["v"].text().strip()
                if k:
                    out.append(f"{k}={val}" if val != "" else k)
            return "\n".join(out)

        def _set(s) -> None:
            for e in list(rows):
                e["w"].setParent(None)
                e["w"].deleteLater()
            rows.clear()
            for tok in str(s or "").replace("\n", " ").split():
                k, _, val = tok.partition("=")
                _add_row(k, val)
            if not rows:
                _add_row()

        self._getters[dest] = _get
        self._setters[dest] = _set
        self._widgets[dest] = box
        _add_row()
        return box

    def _browse(self, edit: QLineEdit, kind: str) -> None:
        start = edit.text().strip() or str(backend.ROOT)
        if kind == "dir":
            path = QFileDialog.getExistingDirectory(self, "Select folder", start)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select file", start)
        if path:
            edit.setText(path)

    def _arg_help_text(self, source_dest: str = "optimizer_type") -> str:
        """Formatted list of the accepted args of the optimizer/scheduler currently
        selected in ``source_dest`` (shown inline by an opthelp panel). Uses
        backend.optimizer_arg_help, which handles both optimizers and schedulers
        (builtin names + dotted paths), torch-free from the cache."""
        g = self._getters.get(source_dest)
        name = str((g() if g else "") or "").strip()
        if not name:
            return tr("Pick an optimizer / scheduler first.")
        try:
            info = backend.optimizer_arg_help(name)
        except Exception as exc:  # noqa: BLE001
            return str(exc)
        if not info or not info.get("ok"):
            return (info or {}).get("note") or f"{name}: no introspectable args"
        lines = []
        if info.get("note"):
            lines.append(info["note"])
        for a in info.get("args", []):
            dv = a.get("default")
            dv = "" if dv is None else f" = {dv}"
            req = "  (required)" if a.get("required") else ""
            lines.append(f"• {a.get('name')}{dv}{req} — {a.get('desc', '')}")
        return "\n".join(lines) or f"{name}: no extra args"

    # ----- schema (auto) field widgets ------------------------------------ #
    def _build_adv_field(self, arg: dict) -> QWidget:
        flag = arg.get("flag")
        help_txt = arg.get("help") or ""
        if arg.get("negatable"):
            combo = _Combo()
            combo.addItems(["default", "on", "off"])
            combo.setToolTip(help_txt)
            self._adv.append(
                (
                    arg,
                    lambda c=combo, f=flag: (
                        {"flag": f, "negatable": True, "tri": c.currentText()}
                        if c.currentText() != "default"
                        else None
                    ),
                )
            )
            self._widgets[arg.get("dest") or ""] = combo
            return combo
        if arg.get("is_bool"):
            cb = QCheckBox()
            cb.setToolTip(help_txt)
            self._adv.append(
                (
                    arg,
                    lambda c=cb, f=flag: (
                        {"flag": f, "is_bool": True, "value": True, "on": True}
                        if c.isChecked()
                        else None
                    ),
                )
            )
            self._widgets[arg.get("dest") or ""] = cb
            return cb
        choices = arg.get("choices") or []
        if choices and {str(x).lower() for x in choices} <= {"true", "false"}:
            # A bare true/false dropdown is just a toggle — render one checkbox
            # (checked → --flag true; unchecked → defer to the config default).
            cb = QCheckBox()
            cb.setToolTip(help_txt)
            self._adv.append(
                (
                    arg,
                    lambda c=cb, f=flag: (
                        {"flag": f, "value": "true", "on": True}
                        if c.isChecked()
                        else None
                    ),
                )
            )
            self._widgets[arg.get("dest") or ""] = cb
            return cb
        if arg.get("choices"):
            # Non-editable (like the negatable/use_cmmd combos) — choices are a fixed
            # argparse set, so no free-text entry; cleaner, matches the other dropdowns.
            combo = _Combo()
            combo.addItem("")
            combo.addItems([str(x) for x in arg["choices"]])
            combo.setToolTip(help_txt)
            self._adv.append(
                (
                    arg,
                    lambda c=combo, f=flag, a=arg: (
                        {
                            "flag": f,
                            "value": c.currentText().strip(),
                            "nargs": a.get("nargs"),
                            "on": True,
                        }
                        if c.currentText().strip()
                        else None
                    ),
                )
            )
            self._widgets[arg.get("dest") or ""] = combo
            return combo
        edit = QLineEdit()
        edit.setToolTip(help_txt)
        self._adv.append(
            (
                arg,
                lambda e=edit, f=flag, a=arg: (
                    {
                        "flag": f,
                        "value": e.text().strip(),
                        "nargs": a.get("nargs"),
                        "on": True,
                    }
                    if e.text().strip()
                    else None
                ),
            )
        )
        self._widgets[arg.get("dest") or ""] = edit
        return edit

    # ----- greying (conflict / dependency) -------------------------------- #
    def _widget_value(self, dest: str) -> object:
        w = self._widgets.get(dest)
        if isinstance(w, QCheckBox):
            return w.isChecked()
        if isinstance(w, QComboBox):
            return w.currentText().strip()
        if isinstance(w, QLineEdit):
            return w.text().strip()
        return None

    def _wire_greying(self) -> None:
        for dest in _GREY_DRIVERS:
            w = self._widgets.get(dest)
            if isinstance(w, QComboBox):
                w.currentTextChanged.connect(lambda *_: self._apply_greying())
            elif isinstance(w, QLineEdit):
                w.textChanged.connect(lambda *_: self._apply_greying())
            elif isinstance(w, QCheckBox):
                w.toggled.connect(lambda *_: self._apply_greying())
        # The train-scope combo gates unet_lr / text_encoder_lr greying.
        if self._scope is not None:
            self._scope.currentIndexChanged.connect(lambda *_: self._apply_greying())
        # Auto-refresh any OPEN args-help panel when its source optimizer/scheduler
        # changes (no collapse + re-expand needed).
        for source in {src for src, _b, _body in getattr(self, "_opthelp_panels", [])}:
            w = self._widgets.get(source)
            if isinstance(w, QComboBox):
                w.currentTextChanged.connect(lambda *_: self._refresh_opthelp())

    def _refresh_opthelp(self) -> None:
        if getattr(self, "_loading", False):
            return  # bulk config-load in progress; refreshed once at the end
        for source, btn, body in getattr(self, "_opthelp_panels", []):
            if btn.isChecked():
                body.setText(self._arg_help_text(source))

    def _apply_greying(self) -> None:
        if getattr(self, "_loading", False):
            return  # bulk config-load in progress; greying runs once at the end
        vals = {d: self._widget_value(d) for d in _GREY_DRIVERS}
        if self._scope is not None:
            vals["__scope__"] = self._scope.currentIndex()
        for target, pred, reason in _GREY_RULES:
            w = self._widgets.get(target)
            if w is not None:
                enabled = bool(pred(vals))
                w.setEnabled(enabled)
                w.setToolTip("" if enabled else tr(reason))  # extra detail on hover
                # Show WHY it's greyed INLINE next to the label (not hover-only).
                lt = self._field_labels.get(target)
                if lt is not None:
                    lbl, base = lt
                    if enabled:
                        lbl.setText(base)
                        lbl.setStyleSheet("")
                    else:
                        lbl.setText(f"{base}  🔒 {tr(reason)}")
                        lbl.setStyleSheet("color:#b07030;")
        for card in getattr(self, "_subset_cards", []):
            fields = card.get("fields", {})
            for col_key, driver in _SUBSET_GREY:
                w = fields.get(col_key)
                if w is not None:
                    w.setEnabled(not _truthy(vals.get(driver)))

    def _build_extra_flags_box(self) -> QGroupBox:
        gb = QGroupBox("Raw extra flags")
        form = QFormLayout(gb)
        edit = QPlainTextEdit()
        edit.setMaximumHeight(70)
        edit.setPlaceholderText("--highvram\n--guidance_scale 1.0")
        self._getters["extra_flags"] = lambda e=edit: e.toPlainText().strip()
        self._setters["extra_flags"] = lambda v, e=edit: e.setPlainText(str(v or ""))
        form.addRow("Anything else", edit)
        return gb

    # ----- subset cards (reference-style collapsible blocks) -------------- #
    def _build_subset_box(self) -> QGroupBox:
        gb = QGroupBox(tr("Subsets"))
        outer = QVBoxLayout(gb)
        add_all = QPushButton(tr("➕ Add all subfolders from a folder…"))
        add_all.clicked.connect(self._subset_add_all_subfolders)
        outer.addWidget(add_all)
        row = QHBoxLayout()
        add_one = QPushButton(tr("➕ Add subset"))
        add_one.clicked.connect(lambda: self._add_subset_card())
        row.addWidget(add_one)
        row.addStretch(1)
        outer.addLayout(row)
        self._subset_cards = []
        self._subset_holder = QWidget()
        self._subset_layout = QVBoxLayout(self._subset_holder)
        self._subset_layout.setContentsMargins(0, 0, 0, 0)
        self._subset_layout.setSpacing(8)
        outer.addWidget(self._subset_holder)
        return gb

    def _add_subset_card(self, values: dict | None = None) -> None:
        """One [[datasets.subsets]] as a collapsible card (reference-style)."""
        values = values or {}
        fields: dict[str, QWidget] = {}
        card = QFrame()
        card.setObjectName("subsetCard")
        cv = QVBoxLayout(card)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(6)
        toggle = QPushButton()
        toggle.setObjectName("subsetHead")
        toggle.setCheckable(True)
        toggle.setChecked(True)
        delete = QPushButton("🗑")
        delete.setObjectName("icon")
        delete.setFixedWidth(40)
        head.addWidget(toggle, 1)
        head.addWidget(delete)
        cv.addLayout(head)

        body = QWidget()
        body.setObjectName("subsetBody")
        fl = QFormLayout(body)
        fl.setContentsMargins(12, 10, 12, 10)

        def _dir(key, label, kind="dir", placeholder=""):
            edit = QLineEdit(str(values.get(key, "") or ""))
            if placeholder:
                edit.setPlaceholderText(placeholder)
            b = QPushButton("📁")
            b.setObjectName("icon")
            b.setFixedWidth(40)
            b.clicked.connect(lambda _=False, e=edit, k=kind: self._browse(e, k))
            rw = QWidget()
            rl = QHBoxLayout(rw)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.addWidget(edit)
            rl.addWidget(b)
            fields[key] = edit
            fl.addRow(tr(label), rw)
            if key == "image_dir":
                edit.textChanged.connect(lambda *_: self._renumber_subsets())

        def _text(key, label, placeholder=""):
            edit = QLineEdit(str(values.get(key, "") or ""))
            if placeholder:
                edit.setPlaceholderText(placeholder)
            fields[key] = edit
            fl.addRow(tr(label), edit)

        _dir("image_dir", "Input image dir", placeholder="Image folder")
        _dir("cache_dir", "Cache dir", placeholder="(auto — shared with primary)")
        _text("num_repeats", "Number of repeats", "1")
        _text("keep_tokens", "Keep tokens", "0")
        cext = _Combo()  # non-editable (use_cmmd style)
        cext.addItems([".txt", ".caption"])
        self._set_combo(cext, values.get("caption_extension") or ".txt")
        fields["caption_extension"] = cext
        fl.addRow(tr("Caption extension"), cext)
        _text("caption_dropout_rate", "Caption dropout rate", "0.0")

        checks = QHBoxLayout()
        for key, label in (
            ("flip_aug", "Flip augment"),
            ("random_crop", "Random crop"),
            ("gradient_checkpointing", "Grad checkpointing"),
            ("is_val", "Validation set (hold out)"),
        ):
            cb = QCheckBox(tr(label))
            cb.setChecked(_truthy(values.get(key)))
            fields[key] = cb
            checks.addWidget(cb)
        checks.addStretch(1)
        fl.addRow(checks)

        # OPTIONAL ARGS — collapsible (batch_size + multi-scale tiers)
        opt_toggle = QPushButton(tr("▸ Optional args"))
        opt_toggle.setObjectName("subOpt")
        opt_toggle.setCheckable(True)
        opt_body = QWidget()
        ofl = QFormLayout(opt_body)
        ofl.setContentsMargins(0, 6, 0, 0)
        bs = QLineEdit(str(values.get("batch_size", "") or ""))
        bs.setPlaceholderText("(dataset default)")
        fields["batch_size"] = bs
        ofl.addRow(tr("Batch size"), bs)
        ti = QLineEdit(str(values.get("tiers", "") or ""))
        ti.setPlaceholderText("e.g. 512,1024")
        fields["tiers"] = ti
        ofl.addRow(tr("Tiers (multi-scale)"), ti)
        opt_body.setVisible(False)
        opt_toggle.toggled.connect(
            lambda on, w=opt_body, b=opt_toggle: (
                w.setVisible(on),
                b.setText(tr("▾ Optional args") if on else tr("▸ Optional args")),
            )
        )
        fl.addRow(opt_toggle)
        fl.addRow(opt_body)

        cv.addWidget(body)
        toggle.toggled.connect(lambda on, w=body: w.setVisible(on))
        toggle.toggled.connect(lambda *_: self._renumber_subsets())

        entry = {"frame": card, "fields": fields, "head": toggle}
        delete.clicked.connect(lambda _=False, e=entry: self._remove_subset_card(e))
        self._subset_cards.append(entry)
        self._subset_layout.addWidget(card)
        self._renumber_subsets()
        if hasattr(self, "_widgets"):
            self._apply_greying()

    def _renumber_subsets(self) -> None:
        for i, e in enumerate(self._subset_cards, 1):
            img = e["fields"]["image_dir"].text().strip()
            name = f"  ·  {Path(img).name}" if img else ""
            arrow = "▾" if e["head"].isChecked() else "▸"
            e["head"].setText(f"{arrow}  {tr('SUBSET')} {i}{name}")

    def _remove_subset_card(self, entry: dict) -> None:
        try:
            self._subset_cards.remove(entry)
        except ValueError:
            return
        entry["frame"].setParent(None)
        entry["frame"].deleteLater()
        self._renumber_subsets()

    def _clear_subset_cards(self) -> None:
        for e in list(self._subset_cards):
            e["frame"].setParent(None)
            e["frame"].deleteLater()
        self._subset_cards = []

    def _subset_add_all_subfolders(self) -> None:
        import os

        parent = QFileDialog.getExistingDirectory(
            self, "Select a parent folder", str(backend.ROOT)
        )
        if not parent:
            return
        subs = sorted(
            os.path.join(parent, d)
            for d in os.listdir(parent)
            if os.path.isdir(os.path.join(parent, d))
        )
        if subs:
            for d in subs:
                self._add_subset_card({"image_dir": d})
        else:
            self._add_subset_card({"image_dir": parent})

    def _collect_subsets(self) -> list[dict]:
        out: list[dict] = []
        for e in self._subset_cards:
            f = e["fields"]
            img = f["image_dir"].text().strip()
            if not img:
                continue
            row: dict = {"image_dir": img}
            for k in (
                "cache_dir",
                "num_repeats",
                "keep_tokens",
                "caption_dropout_rate",
                "batch_size",
                "tiers",
            ):
                w = f.get(k)
                if w is None or not w.isEnabled():
                    continue  # greyed (cache-gated) → inert
                v = w.text().strip()
                if v:
                    row[k] = v
            cext = f.get("caption_extension")
            if cext is not None and cext.currentText().strip():
                row["caption_extension"] = cext.currentText().strip()
            for k in ("flip_aug", "random_crop", "gradient_checkpointing", "is_val"):
                w = f.get(k)
                if w is not None and w.isEnabled() and w.isChecked():
                    row[k] = True
            out.append(row)
        return out

    # ----- utils parent --------------------------------------------------- #
    def _build_utils_parent(self) -> QTabWidget:
        from gui.native.dataset_view import DatasetView

        inner = QTabWidget()
        self._utils_inner = inner
        inner.addTab(DatasetView(), tr("Dataset"))
        inner.addTab(self._scroll(self._build_preprocess_tab()), tr("Preprocess"))
        inner.addTab(self._scroll(self._build_update_tab()), tr("Update"))
        inner.addTab(self._scroll(self._build_autobatch_tab()), tr("Auto-batch"))
        inner.addTab(self._scroll(self._build_masking_tab()), tr("Masking"))
        inner.addTab(self._scroll(self._build_tools_tab()), tr("Tools"))
        return inner

    # ----- diffusion-pipe tools ------------------------------------------- #
    def _tool_path_row(self, store: dict, key: str, kind: str = "file") -> QWidget:
        edit = QLineEdit()
        store[key] = edit
        row = QWidget()
        hb = QHBoxLayout(row)
        hb.setContentsMargins(0, 0, 0, 0)
        hb.addWidget(edit)
        btn = QPushButton("📁")
        btn.setObjectName("icon")
        btn.setFixedWidth(40)
        btn.clicked.connect(lambda _=False, e=edit, k=kind: self._browse(e, k))
        hb.addWidget(btn)
        return row

    def _build_tools_tab(self) -> QWidget:
        w = QWidget()
        vbox = QVBoxLayout(w)
        vbox.addWidget(
            QLabel(
                "diffusion-pipe interop tools (tools/*.py). Output streams to the log."
            )
        )

        # strip-lora-layers
        self._strip: dict[str, object] = {}
        gb1 = QGroupBox("Strip LoRA layers (tools/strip_lora_layers.py)")
        f1 = QFormLayout(gb1)
        f1.addRow("Input LoRA", self._tool_path_row(self._strip, "input"))
        f1.addRow(
            "Output (blank = list only)", self._tool_path_row(self._strip, "output")
        )
        self._strip["strip"] = QLineEdit()
        self._strip["strip"].setPlaceholderText("mlp self_attn llm_adapter")
        f1.addRow("Strip substrings", self._strip["strip"])
        self._strip["dry"] = QCheckBox("dry-run")
        self._strip["force"] = QCheckBox("force overwrite")
        f1.addRow(self._strip["dry"], self._strip["force"])
        b1 = QPushButton("Run strip")
        b1.clicked.connect(self._do_strip_lora)
        f1.addRow(b1)
        vbox.addWidget(gb1)

        # llm-adapter surgery
        self._surg: dict[str, object] = {}
        gb2 = QGroupBox("LLM-adapter surgery (tools/llm_adapter_surgery.py)")
        f2 = QFormLayout(gb2)
        mode = _Combo()
        mode.addItems(["strip", "attach"])
        self._surg["mode"] = mode
        f2.addRow("Mode", mode)
        f2.addRow("Input checkpoint", self._tool_path_row(self._surg, "input"))
        f2.addRow("Donor (attach only)", self._tool_path_row(self._surg, "donor"))
        f2.addRow("Output (blank = default)", self._tool_path_row(self._surg, "out"))
        self._surg["dry"] = QCheckBox("dry-run")
        self._surg["force"] = QCheckBox("force")
        self._surg["extra"] = QCheckBox("allow-empty / replace-existing")
        f2.addRow(self._surg["dry"], self._surg["force"])
        f2.addRow(self._surg["extra"])
        b2 = QPushButton("Run surgery")
        b2.clicked.connect(self._do_llm_surgery)
        f2.addRow(b2)
        vbox.addWidget(gb2)
        vbox.addStretch(1)
        return w

    def _do_strip_lora(self) -> None:
        inp = self._strip["input"].text().strip()
        if not inp:
            QMessageBox.warning(self, "Strip", "Input LoRA is required.")
            return
        argv = ["tools/strip_lora_layers.py", inp]
        out = self._strip["output"].text().strip()
        if out:
            argv.append(out)
        subs = self._strip["strip"].text().split()
        if subs:
            argv += ["--strip", *subs]
        if not out and not subs:
            argv.append("--list-types")
        if self._strip["dry"].isChecked():
            argv.append("--dry-run")
        if self._strip["force"].isChecked():
            argv.append("--force")
        self._run_tool(argv, "strip_lora")

    def _do_llm_surgery(self) -> None:
        mode = self._surg["mode"].currentText()
        inp = self._surg["input"].text().strip()
        if not inp:
            QMessageBox.warning(self, "Surgery", "Input checkpoint is required.")
            return
        argv = ["tools/llm_adapter_surgery.py", mode, inp]
        if mode == "attach":
            donor = self._surg["donor"].text().strip()
            if not donor:
                QMessageBox.warning(self, "Surgery", "Attach needs a donor checkpoint.")
                return
            argv += ["--donor", donor]
        out = self._surg["out"].text().strip()
        if out:
            argv += ["--out", out]
        if self._surg["dry"].isChecked():
            argv.append("--dry-run")
        if self._surg["force"].isChecked():
            argv.append("--force")
        if self._surg["extra"].isChecked():
            argv.append("--replace-existing" if mode == "attach" else "--allow-empty")
        self._run_tool(argv, "llm_adapter")

    def _run_tool(self, argv: list[str], name: str) -> None:
        res = backend.run_tool(argv, name)
        if not res.get("ok"):
            QMessageBox.warning(self, "Tool", str(res.get("error") or res))

    def _build_preprocess_tab(self) -> QWidget:
        w = QWidget()
        vbox = QVBoxLayout(w)
        vbox.addWidget(
            QLabel(
                "Resize → cache (VAE / TE / PE / pooled). Reads configs/preprocess.toml "
                "+ base.toml for paths/target_res. Mutually exclusive with a run; output "
                "streams to the log."
            )
        )
        steps = [
            ("All (resize → cache)", "all"),
            ("Resize", "resize"),
            ("VAE latents", "vae"),
            ("Text-encoder", "te"),
            ("PE features", "pe"),
            ("Pooled TE", "pooled"),
            ("Reconcile (drop stale)", "reconcile"),
        ]
        for label, step in steps:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _=False, s=step: self._do_preprocess(s))
            vbox.addWidget(btn)
        vbox.addStretch(1)
        return w

    def _do_preprocess(self, step: str) -> None:
        res = backend.run_preprocess(step)
        if not res.get("ok"):
            QMessageBox.warning(self, "Preprocess", str(res.get("error") or res))

    def _build_update_tab(self) -> QWidget:
        w = QWidget()
        vbox = QVBoxLayout(w)
        vbox.addWidget(
            QLabel(
                "Update the tool (git pull + uv sync) — datasets/models are gitignored."
            )
        )
        self._update_info = QLabel("—")
        self._update_info.setWordWrap(True)
        vbox.addWidget(self._update_info)
        row = QHBoxLayout()
        check = QPushButton("Check for updates")
        do = QPushButton("Update now")
        check.clicked.connect(self._do_check_update)
        do.clicked.connect(self._do_update)
        row.addWidget(check)
        row.addWidget(do)
        row.addStretch(1)
        vbox.addLayout(row)
        vbox.addStretch(1)
        return w

    def _do_check_update(self) -> None:
        try:
            v = backend.tool_version(fetch=True)
        except Exception as exc:  # noqa: BLE001
            self._update_info.setText(f"error: {exc}")
            return
        self._update_info.setText(
            f"branch {v.get('branch')} · sha {v.get('sha')} · ahead {v.get('ahead')} "
            f"behind {v.get('behind')} · {'up to date' if v.get('up_to_date') else 'update available'}"
            + (f"\n{v.get('note')}" if v.get("note") else "")
        )

    def _do_update(self) -> None:
        try:
            res = backend.update_tool()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Update", str(exc))
            return
        QMessageBox.information(self, "Update", str(res.get("note") or res))

    def _build_autobatch_tab(self) -> QWidget:
        w = QWidget()
        vbox = QVBoxLayout(w)
        vbox.addWidget(
            QLabel(
                "Max-batch search (tasks.py bench-autobatch). Check one or more "
                "resolution tiers (multi-scale) — each is searched. Output → log."
            )
        )
        self._ab: dict[str, object] = {}

        # multi-scale resolution tiers (search each)
        gb_res = QGroupBox("Resolution tiers (multi-scale)")
        rl = QHBoxLayout(gb_res)
        self._ab_res_checks: list[tuple[int, QCheckBox]] = []
        for t in self._options.get("target_res_tiers") or [512, 768, 1024, 1280, 1536]:
            cb = QCheckBox(str(t))
            if t == 1024:
                cb.setChecked(True)
            rl.addWidget(cb)
            self._ab_res_checks.append((int(t), cb))
        rl.addStretch(1)
        vbox.addWidget(gb_res)

        gb = QGroupBox("Search")
        form = QFormLayout(gb)

        def _line(key: str, default: str = "") -> QLineEdit:
            e = QLineEdit(default)
            self._ab[key] = e
            return e

        form.addRow("Max batch", _line("ab_max_batch", "8"))
        form.addRow("Blocks to swap (base)", _line("ab_blocks_to_swap", "0"))
        # blocks_to_swap as a MAX-N search: auto-escalate up to ab_max_swap.
        self._ab_auto_swap = QCheckBox("Auto-escalate blocks_to_swap up to max N")
        self._ab["ab_auto_swap"] = self._ab_auto_swap
        form.addRow(self._ab_auto_swap, _line("ab_max_swap", "26"))
        # activation budget as a MIN search.
        self._ab_auto_budget = QCheckBox("Auto-search activation budget (down to min)")
        self._ab["ab_auto_budget"] = self._ab_auto_budget
        form.addRow(self._ab_auto_budget, _line("ab_budget", "0.1"))
        self._ab_compile = QCheckBox("torch.compile")
        self._ab["ab_compile"] = self._ab_compile
        form.addRow("Compile", self._ab_compile)

        nm = _Combo()  # non-editable (use_cmmd style)
        nm.addItems([str(x) for x in (self._options.get("network_modules") or [])])
        self._set_combo(nm, "networks.lora_anima")
        self._ab["ab_network_module"] = nm
        form.addRow("Network module", nm)
        form.addRow("Network dim", _line("ab_network_dim", "16"))
        form.addRow("Network alpha", _line("ab_network_alpha", "8"))
        form.addRow("network_args", _line("ab_network_args"))
        opt = _Combo()  # non-editable (use_cmmd style)
        opt.addItems([str(x) for x in (self._options.get("optimizers") or [])])
        self._set_combo(opt, "AdamW")
        self._ab["ab_optimizer_type"] = opt
        form.addRow("Optimizer", opt)
        ab_dit = QLineEdit()
        self._ab["ab_dit"] = ab_dit
        dit_row = QWidget()
        hb = QHBoxLayout(dit_row)
        hb.setContentsMargins(0, 0, 0, 0)
        hb.addWidget(ab_dit)
        bd = QPushButton("📁")
        bd.setObjectName("icon")
        bd.setFixedWidth(40)
        bd.clicked.connect(lambda _=False: self._browse(ab_dit, "file"))
        hb.addWidget(bd)
        form.addRow("DiT (blank = config)", dit_row)
        vbox.addWidget(gb)

        btn = QPushButton("Run auto-batch search")
        btn.clicked.connect(self._do_autobatch)
        vbox.addWidget(btn)
        vbox.addStretch(1)
        return w

    def _do_autobatch(self) -> None:
        form: dict = {"ab_res": [t for t, cb in self._ab_res_checks if cb.isChecked()]}
        for key, w in self._ab.items():
            if isinstance(w, QCheckBox):
                form[key] = w.isChecked()
            elif isinstance(w, QComboBox):
                form[key] = w.currentText().strip()
            else:
                form[key] = w.text().strip()
        res = backend.bench_autobatch(form)
        if not res.get("ok"):
            QMessageBox.warning(self, "Auto-batch", str(res.get("error") or res))

    def _build_masking_tab(self) -> QWidget:
        w = QWidget()
        vbox = QVBoxLayout(w)
        gb = QGroupBox("Masking (SAM3 + MIT → merged masks)")
        form = QFormLayout(gb)
        self._mask_sam = QCheckBox()
        self._mask_sam.setChecked(True)
        self._mask_mit = QCheckBox()
        self._mask_mit.setChecked(True)
        self._mit_tt = QLineEdit()
        self._mit_dilate = QLineEdit()
        form.addRow("SAM3", self._mask_sam)
        form.addRow("MIT (text removal)", self._mask_mit)
        form.addRow("MIT text threshold", self._mit_tt)
        form.addRow("MIT dilate", self._mit_dilate)
        vbox.addWidget(gb)
        btn = QPushButton("Run masking")
        btn.clicked.connect(self._do_masking)
        vbox.addWidget(btn)
        vbox.addStretch(1)
        return w

    def _do_masking(self) -> None:
        form = {
            "mask_sam": self._mask_sam.isChecked(),
            "mask_mit": self._mask_mit.isChecked(),
            "mit_text_threshold": self._mit_tt.text().strip(),
            "mit_dilate": self._mit_dilate.text().strip(),
        }
        res = backend.run_masking(form)
        if not res.get("ok"):
            QMessageBox.warning(self, "Masking", str(res.get("error") or res))

    # ----- run panel ------------------------------------------------------ #
    def _build_run_panel(self) -> QWidget:
        panel = QWidget()
        vbox = QVBoxLayout(panel)
        cfg_row = QHBoxLayout()
        btn_load = QPushButton(tr("Load config…"))
        btn_save = QPushButton(tr("Save config…"))
        btn_load.clicked.connect(self._load_config)
        btn_save.clicked.connect(self._save_config)
        cfg_row.addWidget(btn_load)
        cfg_row.addWidget(btn_save)
        cfg_row.addStretch(1)
        # Language selector — switches the whole UI live (en ⇄ 한국어), persisted.
        cfg_row.addWidget(QLabel(tr("Language")))
        self._lang_combo = _Combo()
        self._lang_combo.addItem("English", "en")
        self._lang_combo.addItem("한국어", "ko")
        self._lang_combo.setCurrentIndex(1 if _LANG == "ko" else 0)
        self._lang_combo.currentIndexChanged.connect(
            lambda _i: self._set_language(self._lang_combo.currentData())
        )
        cfg_row.addWidget(self._lang_combo)
        vbox.addLayout(cfg_row)

        vbox.addWidget(QLabel(tr("Command preview")))
        self._preview = QPlainTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setMaximumHeight(120)
        self._preview.setFont(QFont("monospace"))
        vbox.addWidget(self._preview)

        btn_row = QHBoxLayout()
        self._btn_preview = QPushButton(tr("Preview"))
        self._btn_start = QPushButton(tr("▶ Start"))
        self._btn_start.setObjectName("primary")  # the single gold call-to-action
        self._btn_stop = QPushButton(tr("■ Stop"))
        self._btn_monitor = QPushButton(tr("Open monitor"))
        self._btn_preview.clicked.connect(self._do_preview)
        self._btn_start.clicked.connect(self._do_start)
        self._btn_stop.clicked.connect(self._do_stop)
        self._btn_monitor.clicked.connect(self._open_monitor)
        self._btn_monitor.setEnabled(False)
        for b in (
            self._btn_preview,
            self._btn_start,
            self._btn_stop,
            self._btn_monitor,
        ):
            btn_row.addWidget(b)
        vbox.addLayout(btn_row)

        self._status = QLabel(tr("idle"))
        vbox.addWidget(self._status)
        vbox.addWidget(QLabel(tr("Log")))
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("monospace"))
        vbox.addWidget(self._log, 1)

        # Collapsible saved-run queue: the panel sits ABOVE its toggle, so it
        # expands upward (and collapses back down) like a bottom drawer.
        self._queue_panel = self._build_queue_panel()
        self._queue_panel.setVisible(False)
        vbox.addWidget(self._queue_panel)
        self._queue_btn = QPushButton(tr("▲ Queue"))
        self._queue_btn.setCheckable(True)
        self._queue_btn.toggled.connect(self._toggle_queue)
        vbox.addWidget(self._queue_btn)
        return panel

    def _toggle_queue(self, on: bool) -> None:
        self._queue_panel.setVisible(on)
        self._queue_btn.setText("▼ Queue" if on else "▲ Queue")
        if on:
            self._queue_refresh()

    # ----- form <-> dict -------------------------------------------------- #
    def _enabled(self, dest: str) -> bool:
        # A greyed (disabled) field is inert: excluded from the command so its
        # value defers to the config chain (matches the Gradio panel).
        w = self._widgets.get(dest)
        return w is None or w.isEnabled()

    def _collect(self) -> dict:
        form = {
            dest: get() for dest, get in self._getters.items() if self._enabled(dest)
        }
        subsets = self._collect_subsets()
        if subsets:
            form["subsets"] = subsets
        adv = [
            item
            for a, g in self._adv
            if self._enabled(a.get("dest") or "") and (item := g())
        ]
        if self._scope is not None:
            idx = self._scope.currentIndex()
            if idx == 1:
                adv.append(
                    {
                        "flag": "--network_train_unet_only",
                        "is_bool": True,
                        "value": True,
                        "on": True,
                    }
                )
            elif idx == 2:
                adv.append(
                    {
                        "flag": "--network_train_text_encoder_only",
                        "is_bool": True,
                        "value": True,
                        "on": True,
                    }
                )
        if form.get("auto_keep_tokens"):
            # Match the separator the Dataset tab inserts after @artist, so kohya
            # keeps exactly the non-general head per image.
            adv.append(
                {
                    "flag": "--keep_tokens_separator",
                    "value": KEEP_TOKENS_SEPARATOR,
                    "on": True,
                }
            )
        # t5_tokenizer_path is a curated picker but the backend emits only dit/te/
        # vae, so route it through adv (flag --t5_tokenizer_path).
        tok = str(form.get("t5_tokenizer_path") or "").strip()
        if tok and self._enabled("t5_tokenizer_path"):
            adv.append({"flag": "--t5_tokenizer_path", "value": tok, "on": True})
        if adv:
            form["adv"] = adv
        return form

    def _apply(self, form: dict) -> None:
        # Suppress the per-field greying / args-help refresh while bulk-setting (each
        # setter fires signals; doing the cascade N× — and possibly a torch-importing
        # optimizer_arg_help if a help panel is open — froze the GUI on config load).
        # Apply greying ONCE at the end instead.
        self._loading = True
        try:
            for dest, val in form.items():
                setter = self._setters.get(dest)
                if setter:
                    setter(val)
                elif dest in self._widgets:
                    # Schema/advanced fields register no setter (only a getter in
                    # self._adv) — set them onto their widget by dest so a loaded
                    # config's dropdowns / values populate, not just curated fields.
                    self._set_widget_value(dest, val)
            subsets = form.get("subsets")
            if isinstance(subsets, list):
                self._clear_subset_cards()
                for s in subsets:
                    if isinstance(s, dict):
                        self._add_subset_card(s)
        finally:
            self._loading = False
        self._sync_optional_groups()  # auto-enable Sampling/Validation if loaded
        self._apply_greying()  # re-evaluate once after the load changes drivers
        self._refresh_opthelp()  # refresh any open help panel once, now

    def _sync_optional_groups(self) -> None:
        """Tick a collapsible feature group (Sampling/Validation) when a loaded config
        gave any of its fields a value — so loading doesn't silently drop them."""
        for _title, (gb, dests) in self._opt_groups.items():
            if any(_truthy(self._widget_value(d)) for d in dests if d in self._widgets):
                gb.setChecked(True)

    # ----- actions -------------------------------------------------------- #
    def _do_preview(self) -> None:
        try:
            self._preview.setPlainText(" ".join(backend.build_command(self._collect())))
        except Exception as exc:  # noqa: BLE001
            self._preview.setPlainText(f"[preview error] {exc}")

    # ----- required-field validation -------------------------------------- #
    def _required_missing(self) -> list[tuple[str, str]]:
        """(dest, label) of required-but-empty fields. Essentials are always
        required (unless greyed/inert); Sampling/Validation require a cadence only
        when their Enable group is ticked."""
        missing: list[tuple[str, str]] = []

        def _empty(dest: str) -> bool:
            return not str(self._widget_value(dest) or "").strip()

        for dest, label in (
            ("optimizer_type", "Optimizer"),
            ("lr_scheduler_type", "LR scheduler"),
            ("network_module", "Network module"),
        ):
            w = self._widgets.get(dest)
            if w is not None and not w.isEnabled():
                continue  # greyed (e.g. scheduler under a schedule-free optimizer)
            if _empty(dest):
                missing.append((dest, label))

        def _group_on(title: str) -> bool:
            g = self._opt_groups.get(title)
            return bool(g and g[0].isChecked())

        if (
            _group_on("Sampling")
            and _empty("sample_every_n_steps")
            and _empty("sample_every_n_epochs")
        ):
            missing.append(("sample_every_n_epochs", "Sample every N steps/epochs"))
        if (
            _group_on("Validation")
            and _empty("validate_every_n_steps")
            and _empty("validate_every_n_epochs")
        ):
            missing.append(("validate_every_n_epochs", "Validate every N steps/epochs"))
        return missing

    def _highlight_missing(self, missing: list[tuple[str, str]]) -> None:
        for dest in self._highlighted:  # clear previous run's highlights
            w = self._widgets.get(dest)
            if w is not None:
                w.setStyleSheet("")
        self._highlighted = []
        for dest, _label in missing:
            w = self._widgets.get(dest)
            if w is not None:
                w.setStyleSheet("border: 1px solid #e0533a; border-radius: 3px;")
                self._highlighted.append(dest)

    def _do_start(self) -> None:
        self._do_preview()
        missing = self._required_missing()
        if missing:
            self._highlight_missing(missing)
            names = "\n".join(f"• {tr(label)}" for _d, label in missing)
            QMessageBox.warning(
                self,
                tr("Required fields missing"),
                tr("Fill these before starting:") + "\n\n" + names,
            )
            return
        self._highlight_missing([])  # clear any prior red borders
        res = backend.launch(self._collect())
        if not res.get("ok"):
            QMessageBox.critical(self, "Launch failed", str(res.get("error") or res))
            return
        self._log.clear()
        self._log_cache = ""

    def _do_stop(self) -> None:
        res = backend.stop()
        if not res.get("ok"):
            QMessageBox.warning(self, "Stop", str(res.get("error") or res))

    def _open_monitor(self) -> None:
        url = backend.status().get("monitor_url")
        if url:
            QDesktopServices.openUrl(QUrl(url))

    def _load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load config TOML", str(backend.ROOT), "TOML (*.toml);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                # Pass the full set of rendered field dests so schema/advanced args
                # (dropdowns + values) populate their fields instead of extra_flags.
                form = load_toml_to_form(f.read(), known_dests=set(self._widgets))
            self._apply(form)
            self._do_preview()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load failed", str(exc))

    def _save_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save config TOML", str(backend.ROOT / "config.toml"), "TOML (*.toml)"
        )
        if not path:
            return
        try:
            text = save_form_to_toml(self._collect())
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", str(exc))

    # ----- polling -------------------------------------------------------- #
    def _poll(self) -> None:
        st = backend.status()
        running = bool(st.get("running"))
        self._btn_start.setEnabled(not running)
        self._btn_stop.setEnabled(running)
        self._btn_monitor.setEnabled(bool(st.get("monitor_url")))
        if running:
            self._status.setText(
                f"running · pid {st.get('pid')} · {int(st.get('elapsed') or 0)}s"
            )
        elif st.get("returncode") is not None:
            self._status.setText(f"finished · exit {st.get('returncode')}")
        else:
            self._status.setText("idle")
        lines = backend.log_tail(400).get("lines") or []
        text = "\n".join(lines)
        # Only touch the document when it actually changed. Re-setting it every poll
        # re-rendered and yanked the scrollbar to the TOP whenever the user had
        # scrolled up to read history (the "log jumps up" bug). Preserve position:
        # pin to bottom only if already there, otherwise keep the user's spot.
        if text != getattr(self, "_log_cache", None):
            sb = self._log.verticalScrollBar()
            at_bottom = sb.value() >= sb.maximum() - 4
            prev = sb.value()
            self._log.setPlainText(text)
            self._log_cache = text
            sb.setValue(sb.maximum() if at_bottom else min(prev, sb.maximum()))


# ──────────────────────────────────────────────────────────────────────────── #
# Theme — "Gemini" near-black + gold accent, modelled on the Image-viewer Vue UI
# (frontend/src/style.css). Dark QPalette for default widget surfaces + a QSS pass
# for rounding / accent / inputs / scrollbars so the whole panel reads as one sleek
# dark surface instead of the default OS grey.
# ──────────────────────────────────────────────────────────────────────────── #
_C = {
    "bg": "#0A0A0A",  # window / tab pages
    "card": "#141414",  # group boxes
    "input": "#1C1C1C",  # inputs / buttons
    "raised": "#242424",  # hover
    "border": "#2A2A2A",
    "accent": "#FACC15",  # Gemini gold
    "accent_hi": "#FFE04A",
    "text": "#EDEDED",
    "muted": "#9A9A9A",
    "faint": "#5A5A5A",
}

_QSS = """
* {{ font-family: 'Pretendard','Segoe UI','Inter',sans-serif; font-size: 13px; outline: none; }}
QMainWindow, QDialog {{ background: {bg}; }}
QToolTip {{ background: {input}; color: {text}; border: 1px solid {border};
           padding: 5px 8px; border-radius: 6px; }}
QLabel#fadeTip {{ background: {input}; color: {text}; border: 1px solid {border};
                 border-radius: 8px; padding: 7px 10px; font-size: 12px; }}

QSplitter::handle {{ background: {bg}; }}
QSplitter::handle:horizontal {{ width: 6px; }}

/* Tabs */
QTabWidget::pane {{ border: 1px solid {border}; border-radius: 12px; background: {bg};
                   top: -1px; padding: 4px; }}
QTabBar {{ qproperty-drawBase: 0; background: transparent; }}
QTabBar::tab {{ background: transparent; color: {muted}; padding: 8px 16px; margin: 2px;
               border: 1px solid transparent; border-radius: 8px; font-weight: 600; }}
QTabBar::tab:hover {{ color: {text}; background: {card}; }}
QTabBar::tab:selected {{ color: #000; background: {accent}; }}

/* Cards */
QGroupBox {{ background: {card}; border: 1px solid {border}; border-radius: 12px;
            margin-top: 16px; padding: 14px 12px 12px 12px; font-weight: 600; color: {text}; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 14px;
                   padding: 2px 8px; color: #C8C8C8; background: {card};
                   border-radius: 6px; font-size: 11px; font-weight: 700; }}

QLabel {{ color: {muted}; background: transparent; }}

/* Buttons */
QPushButton {{ background: {input}; color: {text}; border: 1px solid {border};
              border-radius: 8px; padding: 7px 14px; font-weight: 600; }}
QPushButton:hover {{ background: {raised}; border-color: {faint}; color: #fff; }}
QPushButton:pressed {{ background: {bg}; }}
QPushButton:disabled {{ color: {faint}; background: #131313; border-color: #1c1c1c; }}
QPushButton:checked {{ background: {accent}; color: #000; border-color: {accent}; }}
QPushButton#primary {{ background: {accent}; color: #000; border: none; font-weight: 800;
                      padding: 8px 18px; }}
QPushButton#primary:hover {{ background: {accent_hi}; }}
QPushButton#primary:disabled {{ background: #3a3413; color: #777; }}
/* Compact icon buttons (📁 pickers): tight padding so the glyph isn't clipped by
   the default button padding inside their fixed width. */
QPushButton#icon {{ padding: 4px 0; font-size: 15px; }}
/* Subset cards (reference-style collapsible blocks) */
QFrame#subsetCard {{ background: {card}; border: 1px solid {border}; border-radius: 10px; }}
QWidget#subsetBody {{ background: transparent; }}
QPushButton#subsetHead {{ background: {accent}; color: #000; border: none; border-radius: 9px;
                         text-align: left; padding: 9px 14px; font-weight: 800; }}
QPushButton#subsetHead:hover {{ background: {accent_hi}; }}
QPushButton#subsetHead:checked {{ background: {accent}; color: #000; }}
QPushButton#subOpt {{ background: transparent; border: 1px solid {border}; border-radius: 7px;
                     color: {muted}; text-align: left; padding: 6px 10px; font-weight: 600; }}
QPushButton#subOpt:hover {{ color: {text}; border-color: {faint}; }}
QPushButton#subOpt:checked {{ background: transparent; color: {text}; border-color: {faint}; }}
QLabel#optHelpBody {{ background: {input}; color: {text}; border: 1px solid {border};
                     border-radius: 8px; padding: 8px 10px; }}
QLabel#argDesc {{ color: {muted}; background: transparent; font-size: 12px; }}

/* Inputs */
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox {{
    background: {input}; color: {text}; border: 1px solid {border}; border-radius: 8px;
    padding: 6px 10px; selection-background-color: {accent}; selection-color: #000; }}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {accent}; background: {card}; }}
QLineEdit:disabled, QPlainTextEdit:disabled {{ color: #3f3f3f; background: #0d0d0d;
    border: 1px dashed #2c2c2c; }}
QLineEdit::placeholder {{ color: {faint}; }}

/* Combo */
QComboBox {{ background: {input}; color: {text}; border: 1px solid {border};
            border-radius: 8px; padding: 6px 10px; }}
QComboBox:hover {{ border-color: {faint}; }}
QComboBox:focus, QComboBox:on {{ border-color: {accent}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{ width: 0; height: 0; border-left: 4px solid transparent;
    border-right: 4px solid transparent; border-top: 5px solid {muted}; margin-right: 8px; }}
QComboBox QAbstractItemView {{ background: {input}; color: {text}; border: 1px solid {border};
    border-radius: 8px; selection-background-color: {accent}; selection-color: #000;
    outline: none; padding: 4px; }}
/* Clearly-disabled look (a dashed, dimmed, darker field) so a greyed-out control is
   obviously inactive — hover shows the reason. */
QComboBox:disabled {{ background: #0d0d0d; color: #3f3f3f; border: 1px dashed #2c2c2c; }}
QComboBox::down-arrow:disabled {{ border-top-color: #3a3a3a; }}

/* Checkboxes */
QCheckBox, QRadioButton {{ color: {text}; spacing: 8px; background: transparent; }}
QCheckBox:disabled, QRadioButton:disabled {{ color: {faint}; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 16px; height: 16px;
    border: 1px solid {border}; border-radius: 4px; background: {input}; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {accent}; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {accent}; border-color: {accent}; }}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    background: #0d0d0d; border: 1px dashed #2c2c2c; }}

/* Tables / lists */
QTableWidget, QListWidget {{ background: {bg}; alternate-background-color: {card};
    color: {text}; border: 1px solid {border}; border-radius: 8px;
    gridline-color: {border}; outline: none; }}
QTableWidget::item, QListWidget::item {{ padding: 4px; }}
QTableWidget::item:selected, QListWidget::item:selected {{ background: {accent}; color: #000; }}
QHeaderView::section {{ background: {input}; color: {muted}; padding: 6px 8px; border: none;
    border-right: 1px solid {border}; border-bottom: 1px solid {border}; font-weight: 700; }}
QTableCornerButton::section {{ background: {input}; border: none; }}

/* Scroll */
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {border}; border-radius: 5px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: {faint}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {border}; border-radius: 5px; min-width: 28px; }}
QScrollBar::handle:horizontal:hover {{ background: {faint}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
""".format(**_C)


class _Combo(QComboBox):
    """A combo that does NOT change its value on mouse-wheel scroll. The default Qt
    behavior silently switches a dropdown when you scroll the page past it (very easy
    to mis-set in a long form); here the wheel is ignored so it bubbles to the parent
    scroll area and the page scrolls instead. The popup list still scrolls when open."""

    def wheelEvent(self, e):  # noqa: N802 (Qt signature)
        e.ignore()


class _FadeTooltip(QObject):
    """App-wide event filter that replaces the default abrupt QToolTip with a small
    frameless label that *fades in* near the cursor (스르륵), instead of snapping
    open. Fully guarded — any failure falls back to no tooltip, never a crash."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._label = QLabel(None)
        self._label.setObjectName("fadeTip")
        self._label.setWindowFlags(
            Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowTransparentForInput
        )
        self._label.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self._label.setWordWrap(True)
        self._label.setMaximumWidth(420)
        self._eff = QGraphicsOpacityEffect(self._label)
        self._label.setGraphicsEffect(self._eff)
        self._anim = QPropertyAnimation(self._eff, b"opacity", self)
        self._anim.setDuration(150)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._hide = QTimer(self)
        self._hide.setSingleShot(True)
        self._hide.timeout.connect(self._label.hide)

    def eventFilter(self, obj, event):  # noqa: N802 (Qt signature)
        try:
            et = event.type()
            if et == QEvent.Type.ToolTip:
                text = obj.toolTip() if hasattr(obj, "toolTip") else ""
                if text:
                    self._label.setText(text)
                    self._label.adjustSize()
                    gp = event.globalPos()
                    self._label.move(gp.x() + 14, gp.y() + 18)
                    self._eff.setOpacity(0.0)
                    self._label.show()
                    self._anim.stop()
                    self._anim.start()
                    self._hide.start(8000)
                    return True  # suppress the default snap-open tooltip
                self._label.hide()
            elif et in (QEvent.Type.Leave, QEvent.Type.WindowDeactivate):
                self._label.hide()
        except Exception:
            return False
        return False


def _apply_theme(app: QApplication) -> None:
    """Dark 'Gemini' palette + QSS. Palette covers default surfaces (so nothing flashes
    OS-grey); QSS does rounding / gold accent / inputs / scrollbars."""
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(_C["bg"]))
    pal.setColor(QPalette.WindowText, QColor(_C["text"]))
    pal.setColor(QPalette.Base, QColor(_C["input"]))
    pal.setColor(QPalette.AlternateBase, QColor(_C["card"]))
    pal.setColor(QPalette.Text, QColor(_C["text"]))
    pal.setColor(QPalette.Button, QColor(_C["input"]))
    pal.setColor(QPalette.ButtonText, QColor(_C["text"]))
    pal.setColor(QPalette.ToolTipBase, QColor(_C["input"]))
    pal.setColor(QPalette.ToolTipText, QColor(_C["text"]))
    pal.setColor(QPalette.Highlight, QColor(_C["accent"]))
    pal.setColor(QPalette.HighlightedText, QColor("#000000"))
    pal.setColor(QPalette.PlaceholderText, QColor(_C["faint"]))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(_C["faint"]))
    pal.setColor(QPalette.Disabled, QPalette.WindowText, QColor(_C["faint"]))
    app.setPalette(pal)
    app.setStyle("Fusion")  # consistent base across OSes; QSS refines it
    app.setStyleSheet(_QSS)
    # Kill the inconsistent open animations (some combos slid open, some didn't) — the
    # user prefers none. Unifies every dropdown to instant open.
    for _eff in (
        Qt.UIEffect.UI_AnimateCombo,
        Qt.UIEffect.UI_AnimateMenu,
        Qt.UIEffect.UI_AnimateTooltip,
        Qt.UIEffect.UI_FadeTooltip,
    ):
        try:
            QApplication.setEffectEnabled(_eff, False)
        except Exception:  # noqa: BLE001
            pass


def run() -> None:
    """Create the QApplication and show the main window (blocking)."""
    app = QApplication.instance() or QApplication(sys.argv)
    _apply_theme(app)
    # Smooth fade-in tooltips (스르륵) instead of the default snap-open. Kept on the
    # app so it lives as long as the app (a local would be GC'd).
    app._fade_tip = _FadeTooltip(app)
    app.installEventFilter(app._fade_tip)
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    run()
