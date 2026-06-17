# GUI 회색처리 · 연관인자 정리

네이티브 GUI(`gui/native/app.py`)의 **강제 회색처리(forced-interaction greying)** 규칙과
**연관인자 모음**을 정리한다. 출처는 `kohya / 67372a / anima_lora` 3개 저장소 교차
레퍼런스이나, **이 저장소(anima_lora)에 실제로 존재하고 코드가 소비하는 인자만** 규칙으로
넣는다(나머지는 §3에 "적용 안 됨"으로 분리). 모든 조건은 doc 주장이 아니라 **소비 코드**로
근거를 확인했다.

회색처리 = 어떤 인자(driver)의 값이 다른 인자(target)를 무의미하게 만들면, target을
비활성화하고 라벨에 `🔒 사유`를 **인라인**으로 표시하며, 실행 커맨드에서도 제외한다
(`MainWindow._apply_greying` / `_GREY_RULES` / `_GREY_DRIVERS`).

## 1. 강제 회색처리 규칙 (`_GREY_RULES`)

| target (비활성 대상) | 활성 조건 (driver) | 근거 코드 |
|---|---|---|
| `huber_c`, `huber_schedule`, `huber_scale` | `loss_type ∈ {huber, smooth_l1}` | `library/training/losses.py` (huber 항만 huber_c/scale 사용) |
| `sigmoid_scale`, `sigmoid_bias` | `timestep_sampling ∈ {sigmoid, shift, flux_shift}` | `library/runtime/noise.py:104-122` (세 분기만 sigmoid_scale·sigmoid_bias 읽음) |
| `discrete_flow_shift` | `timestep_sampling == shift` | `noise.py:111-116` (shift 분기 전용; **flux_shift는 해상도 mu 사용**, 미사용) |
| `logit_mean`, `logit_std` | `weighting_scheme == logit_normal` | `noise.py:124-130` → `compute_density_for_timestep_sampling` |
| `mode_scale` | `weighting_scheme == mode` | 동일 (SD3 density 경로, mode 전용) |
| `ip_noise_gamma_random_strength` | `ip_noise_gamma > 0` | `noise.py:159-166` (`if args.ip_noise_gamma:` 안에서만 읽음) |
| `dim_from_weights` | `network_weights` 지정됨 | `cli_args.py` help: "determine dim from network_weights" |
| `lr_scheduler_num_cycles` | `lr_scheduler_type ∈ {cosine_with_restarts, cosine_with_min_lr, warmup_stable_decay}` | `library/training/schedulers.py:210-260` |
| `lr_scheduler_power` | `lr_scheduler_type == polynomial` | `schedulers.py:219-226` (polynomial 전용) |
| `lr_scheduler_type` | `use_constantcosine` off **및** 옵티마이저가 schedule-free 아님 | constantcosine이 스케줄러 대체 / schedule-free는 자체 스케줄 |
| `lr_warmup_steps` | 옵티마이저가 schedule-free 아님 | schedule-free 옵티마이저는 warmup 무시 |
| `use_vae_cache`, `use_text_cache` | `auto_preprocess` off | Auto-전처리가 캐시 빌드+사용을 관리 |
| `unet_lr` | train scope ≠ TE-only | TE-only 스코프에선 DiT LR 미사용 |
| `text_encoder_lr` | train scope ≠ UNet-only | UNet-only 스코프에선 TE LR 미사용 |

서브셋 카드 레벨(`_SUBSET_GREY`): VAE 캐시 on → `random_crop` 비활성, TE 캐시 on →
`caption_dropout_rate` 비활성(라이브 인코딩 전용 증강은 캐시가 있으면 무의미).

## 2. 연관인자 모음 (함께 움직이는 그룹)

규칙으로 회색처리하지는 않더라도 **세트로 이해/설정해야 하는** 인자 묶음.

- **σ 샘플링**: `timestep_sampling` → `{sigmoid_scale, sigmoid_bias}`(sigmoid/shift/flux_shift)
  · `discrete_flow_shift`(shift) · `{weighting_scheme, logit_mean, logit_std, mode_scale}`(sigma 분기).
  ANIMA 표준 = `sigmoid + sigmoid_scale≈1.3`, discrete_flow_shift 미사용.
- **Huber loss**: `loss_type=huber|smooth_l1` → `huber_c` · `huber_schedule` · `huber_scale`.
- **LR 스케줄러 shape**: `lr_scheduler_type`(이름) → `lr_scheduler_num_cycles`(restart/min-lr/WSD)
  · `lr_scheduler_power`(polynomial) · `lr_warmup_steps` · `lr_scheduler_args`.
- **캐시 ↔ 증강(상호배타)**: `use_vae_cache` ↔ `color_aug`/`random_crop`(매 에폭 재인코딩 필요);
  `flip_aug`는 `_flip.npz`로 부분 양립. `use_text_cache` ↔ TE LoRA 학습 / caption shuffle
  (단 ANIMA는 `caption_dropout_rate`만 캐시와 병용 가능, `use_shuffled_caption_variants`로 셔플 우회).
- **재개/병합**: `network_weights` → `dim_from_weights`; `base_weights` → `base_weights_multiplier`.
- **학습 스코프**: train scope 콤보 → `network_train_unet_only`/`network_train_text_encoder_only`
  → `unet_lr`/`text_encoder_lr` 사용처 결정.
- **입력 섭동 노이즈**: `ip_noise_gamma` → `ip_noise_gamma_random_strength`.
- **Auto-전처리**: `auto_preprocess` → `use_vae_cache`/`use_text_cache`(관리).

## 3. 이 엔진에 적용 안 되는 항목 (의도적 제외)

레퍼런스 doc에는 있으나 **anima_lora 인자 표면에 없거나(MISSING) flow-matching에 무의미**해서
회색처리 대상이 아니다. 향후 재추가 금지.

- **flow-matching 부적합 / SD 잔재**: `noise_offset`·`adaptive_noise_scale`·
  `multires_noise_*`(MISSING), `v_parameterization`·`zero_terminal_snr`·
  `scale_v_pred_loss_like_noise_pred`(MISSING), `min_snr_gamma`·`debiased_estimation_loss`
  (인자 선언·메타데이터 기록은 되지만 **손실 계산에서 소비되지 않는 inert 잔재** — repo 전체
  grep상 적용 지점 없음; SNR 기반이라 flow-matching에 무의미. field-간 강제관계가 아니라
  "메서드 부적합"이라 greying 모델엔 안 맞아 GUI 도움말에만 "미적용" 명시).
- **AR 버킷팅**: `enable_bucket`·`bucket_no_upscale`·`min/max_bucket_reso`(MISSING) —
  anima는 constant-token 네이티브 버킷팅 사용.
- **LyCORIS / LoKr 계열**: `algo`·`factor`·`full_matrix`·`use_tucker`·`decompose_both`
  (anima_lora 미지원 — linear-delta 설계). full-matrix LoKr는 kohya/67372a 전용.
- **67372a 전용 스케줄러**: REX/RAWR 등 — 이 저장소엔 없음(`lr_scheduler_type` 커스텀
  dotted-path로만 임의 모듈 주입 가능).
