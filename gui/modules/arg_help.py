# -*- coding: utf-8 -*-
"""Korean per-field help (ported from the web GUI ARG_HELP_KO) → kohya GUI
component `info=` tooltips. Keyed by the form field name; reg() attaches it.
"""

ARG_HELP = {
    "attn_mode": "attention 구현 방식 선택 (sageattn은 추론 전용, --sdpa 무시)",
    "blocks_to_swap": "forward/backward 중 swap할 블록 수 [실험적]",
    "caption_tag_dropout_rate": "쉼표 구분 토큰 드롭아웃 비율 (0.0~1.0)",
    "gradient_accumulation_steps": "업데이트 전 누적할 gradient step 수",
    "gradient_checkpointing": "gradient checkpointing 활성화",
    "huber_c": "Huber 손실 감쇠 파라미터 (기본 0.1)",
    "huber_schedule": "Huber 손실 스케줄링 방식 (기본 snr)",
    "log_every_n_steps": "N 글로벌 스텝마다만 스텝 메트릭 출력 (기본: 1)",
    "logit_mean": "logit_normal 가중 방식의 평균",
    "logit_std": "logit_normal 가중 방식의 표준편차",
    "loss_type": "손실 함수 종류 (L1/L2/Huber/smooth L1/pseudo-Huber, 기본 L2)",
    "lr_scheduler_type": "LR 스케줄러 (dotted path; 보통 자체 warmup → lr_warmup_steps 0). schedule-free 옵티마이저는 자동 우회",
    "max_grad_norm": "최대 gradient 노름 (0=클리핑 없음)",
    "mixed_precision": "혼합 정밀도 (Anima는 bf16, fp16/no는 미검증)",
    "network_train_unet_only": "U-Net만 학습",
    "optimizer_type": "옵티마이저 (kohya 빌트인 + vendored zoo ~89종; 친근한 이름 CAME/Prodigy/ADOPT… 또는 dotted path)",
    "output_config": "커맨드라인 인자를 .toml 파일로 출력",
    "output_dir": "학습 모델 출력 디렉터리",
    "qwen3_max_token_length": "Qwen3 토크나이저 최대 토큰 길이 (기본: 512)",
    "resume": "학습 재개용 저장된 state",
    "sample_prompts": "샘플 이미지 생성용 프롬프트 파일",
    "save_every_n_epochs": "N epoch마다 체크포인트 저장",
    "save_precision": "저장 정밀도 (None=학습 weight dtype)",
    "save_state": "모델 저장 시 학습 state(optimizer 등) 함께 저장",
    "sigmoid_scale": "sigmoid 타임스텝 샘플링 스케일 (기본: 1.0)",
    "t_max": "학습 sigma 범위 상한 — flow-matching sigma 0.0~1.0 (kohya max_timestep 아님, 0~1000 아님! 기본 1.0)",
    "t_min": "학습 sigma 범위 하한 — flow-matching sigma 0.0~1.0 (kohya min_timestep 아님, 0~1000 아님! 기본 0.0)",
    "target_res": "데이터셋 전처리에 쓴 멀티스케일 constant-token tier (512 768 896 1024 1280 1536), 디스크상 모든 tier 나열 필요",
    "timestep_sampling": "타임스텝 샘플링 방식 (기본: sigmoid)",
    "torch_compile": "torch.compile 사용 (PyTorch 2.0+ 필요)",
    "use_shuffled_caption_variants": "TE 캐시의 전처리된 캡션 셔플 변형을 샘플마다 무작위 사용",
    "use_shuffled_caption_variants_only": "원본 v0 제외, 셔플된 v1~v{N-1}만 균등 샘플링",
    "use_text_cache": "TE 출력을 디스크에 캐시 후 학습 시 읽기 (false면 라이브 인코딩)",
    "use_vae_cache": "VAE 잠재값을 디스크에 캐시 후 학습 시 읽기 (false면 라이브 인코딩)",
    "weighting_scheme": "타임스텝 분포 가중 방식 (기본: uniform)"
}
