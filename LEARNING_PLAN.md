# nanochat로 LLM을 바닥부터 이해하기 — 학습 계획

> 데이터 파이프라인부터 트랜지스터·전자 수준까지, nanochat 코드베이스를 척추 삼아 LLM의 작동원리를 바닥부터 학습하는 13~15주 커리큘럼.
>
> **이 문서 사용법**: 각 Stage는 *목표 → 왜 여기인가 → 읽을 코드 → 핵심 개념 → 하드웨어 깊이 → 직접 구현 → 자가시험 → 기간* 순서. 위에서 아래로 진행하되, 막히면 "왜 여기인가"를 다시 읽어 동기를 회복할 것. 맨 아래 **진행 추적** 체크박스에 완료 표시.
>
> *line 인용은 현재 커밋 기준(2026-06-30 검증)이며 코드 변경 시 몇 줄 어긋날 수 있음. `gpt.py`·`optim.py`·`dataset.py`·`flash_attention.py`는 이번에 직접 정독 검증, 나머지는 함수/개념명 위주로 표기.*

---

## 0. 설계 철학 (왜 이 순서인가)

세 원칙으로 순서를 정했다.

1. **"동작을 먼저 보고, 그다음 바닥까지 판다."** 순수 bottom-up(트랜지스터→…→챗봇)은 동기를 잃는다. `BOS`, `bits-per-byte`, `CORE` 같은 용어가 *어디에 쓰이는 부품인지* 모른 채 외우게 된다. 그래서 Stage 0에서 전체 파이프라인을 한 번 돌려 **끝그림**을 먼저 본다.
2. **데이터 파이프라인이 실질적 출발점.** "데이터셋 세팅 → 코퍼스 → 토큰 임베딩" 흐름을 따른다. 데이터가 모델로 *흘러드는 방향*을 그대로 따라가면 자연스럽게 의존성 순서가 된다.
3. **하드웨어 깊이를 한 번에 쏟지 않고 "그 개념이 절실해지는 코드 줄"에 못박는다.** 토큰 바이트를 버퍼에 담을 때 → 메모리 셀·전자. `F.linear`를 읽을 때 → 부동소수 비트·가산기·텐서코어. attention/KV-cache를 만질 때 → 메모리 계층·roofline. 학습 한 스텝을 추적할 때 → "Python 한 줄 → 전자"의 전체 스택. **추상이 아니라 "지금 이 줄 때문에" 배우게** 된다.

> ⚠️ **"단어 임베딩"에 대한 정정**: nanochat에는 *단어* 임베딩이 없다. BPE **토큰** 임베딩(`wte`, `gpt.py:170`)이 있고, 토큰은 단어가 아니라 "바이트열의 자주 등장하는 조각"이다. 이 구분 자체가 Stage 2의 핵심 깨달음이다.

---

## 1. 전체 순서 한눈에 보기

| Stage | 주제 | 핵심 파일 | 하드웨어 깊이(여기서 채움) | 기간 |
|---|---|---|---|---|
| **0** | 오리엔테이션: 전체 한 번 돌리고 지도 그리기 | `README`, `runs/speedrun.sh`, `runs/runcpu.sh`, `nanochat/common.py` | (사다리 빈 지도만) | 반나절~1일 |
| **1** | 데이터 파이프라인: 텍스트 → (input,target) 배치 | `dataset.py`, `dataloader.py`, `repackage_data_reference.py` | 비트의 물리적 실체: 전자·MOSFET·DRAM셀·pinned DMA | 1주 |
| **2** | 토크나이저: 바이트→merge→vocab BPE | `tokenizer.py`, `scripts/tok_train.py`, `scripts/tok_eval.py` | UTF-8 인코딩, 정수 표현, 정보이론(bit/nat) | 1주 |
| **3** | GPT forward: 토큰 → logits | `gpt.py` | **부동소수 비트 + 가산기→곱셈기→MAC→FMA→텐서코어→SIMT** | **2.5~3주** |
| **4** | 정밀도·커널: dtype/FP8/Flash Attention | `common.py`, `fp8.py`, `flash_attention.py` | fp8 비트, 메모리 계층, **roofline·arithmetic intensity** | 1.5주 |
| **5** | 학습 루프·옵티마이저: forward/backward/step + Muon | `scripts/base_train.py`, `optim.py` | **"Python 한 줄 → 전자"의 11계단 풀스택 종합** | **2.5~3주** |
| **6** | 체크포인트·리포트: 저장/재구성 인프라 | `checkpoint_manager.py`, `report.py` | meta device, per-rank 샤딩 복습 | 반나절 |
| **7** | 베이스 평가: bits-per-byte 심화 + DCLM CORE | `loss_eval.py`, `core_eval.py`, `scripts/base_eval.py` | fp32 누산, all_reduce 순서불변성 | 1주 |
| **8** | 추론 엔진: KV-cache prefill/decode + 도구사용 | `engine.py`, `execution.py`, `scripts/chat_cli.py` | **왜 decode는 메모리 대역폭에 묶이는가(roofline 재방문)** | 1.5주 |
| **9** | SFT: assistant 토큰만 감독하는 loss masking | `scripts/chat_sft.py`, `tasks/*` | int32/int64 dtype, prefetch overlap | 1주 |
| **10** | RL(GRPO) + 채팅 평가 | `scripts/chat_rl.py`, `scripts/chat_eval.py` | KV-cache가 rollout을 가능케 함 | 1.5주 |
| **★** | **캡스톤: 무참조 재구현** | (전체) | 양방향 사다리 설명 | 1.5~2주 |

**총 ~13~15주** (주당 10~15시간 기준). 빠르게 가려면 Stage 6/7/10을 압축해 ~10주로 줄일 수 있다.

---

## 2. 왜 이 순서인가 (논리적 근거)

순서 = **의존성 위상정렬 + 인지부하 관리 + 동기부여**의 절충이다. 데이터가 모델로 흘러드는 방향 그대로:

```
데이터로더(1) → 토큰의 정체(2) → forward로 logits(3) → 그 matmul이 실리콘에서 도는 법(4)
   → loss로 가중치 갱신하는 backward(5) → 저장/재구성(6) → 베이스 품질 측정(7)
   → 학습된 모델로 토큰 생성(8) → 채팅 어시스턴트로 다듬기: SFT(9) → RL(10)
```

이는 정확히 `runs/speedrun.sh`의 실제 실행 순서(`dataset → tok_train → base_train → base_eval → chat_sft → chat_eval`)와 일치한다. **각 단계가 다음 스크립트를 읽기 위한 선수지식**이 되도록 정렬돼 있다.

비자명한 배치 결정 몇 가지:
- **정밀도(4)를 학습 루프(5)보다 앞에**: bf16/fp8/`torch.compile` 커널이 학습 루프의 *전제*이기 때문.
- **체크포인트(6)를 학습 직후·평가 직전에**: `load_model`이 평가(7)·추론(8)·SFT(9)·RL(10) 전부의 **공통 진입점**이라서.
- **추론(8)을 SFT(9)·RL(10)보다 앞에**: chat eval과 RL rollout이 모두 `Engine`에 의존.

---

## 3. 하드웨어 "깊이 사다리" — 11계단을 어디서 채우나

"0과 1, 전기 작동까지"는 이 사다리로 구현된다. **한 번에 다 배우지 않고** 단계마다 한두 칸씩 채운다.

```
[높은 추상]
 11. PyTorch 연산 (F.linear, model(x))         ← Stage 3에서 시작
 10. CPython bytecode (LOAD_FAST/CALL)          ← Stage 5
  9. ATen dispatch / torch.compile FX graph     ← Stage 5
  8. CUDA 커널 (PTX → SASS 기계어)               ← Stage 5
  7. 메모리 계층/바이트 (HBM·L2·SRAM·register)   ← Stage 1(저장) + Stage 4(대역폭)
  6. 메모리 셀 (DRAM 1T1C, SRAM 6T)             ← Stage 1
  5. 텐서코어 / SIMT warp (MMA, 32 lane)         ← Stage 3
  4. 가산기·곱셈기·MAC·FMA                       ← Stage 3
  3. 논리 게이트 (AND/XOR/full-adder)            ← Stage 3
  2. 트랜지스터 (MOSFET = 전압으로 여닫는 스위치) ← Stage 1·3
  1. 전자/전압 (0과 1의 물리적 실체, 노이즈마진)  ← Stage 1
[전기]
```

**핵심 통찰**: matmul 한 번(`F.linear`)을 따라가면 "11→1"의 거의 전 계단을 한 줄에서 만난다. 그래서 Stage 3이 가장 깊은 하드웨어 단계이고 기간도 가장 길다.

---

## 4. 단계별 상세

### Stage 0 — 오리엔테이션 (반나절~1일)
- **목표**: 전체가 어떻게 연결되는지 한 번 돌려보고, 위 11계단 사다리를 *빈칸으로* 그려둔다.
- **하라**:
  - `uv sync --extra cpu && source .venv/bin/activate`
  - GPU 있으면 README의 짧은 `d12` 실험으로 끝그림 관찰. **GPU 없으면 각 스크립트를 1 step만** 돌려 입출력만 확인(완주 목표 ❌, *관찰* 목표 ⭕).
  - `python -c "from nanochat.common import COMPUTE_DTYPE, COMPUTE_DTYPE_REASON; print(COMPUTE_DTYPE, COMPUTE_DTYPE_REASON)"` → 내 GPU의 SM 버전으로 왜 bf16/fp32인지 설명.
- **자가시험**: `speedrun.sh`의 단계를 각각 "무엇을 입력받아 무엇을 산출하는가"로 한 문장씩 말할 수 있는가?

---

### Stage 1 — 데이터 파이프라인 (1주) — *실질적 출발점*
- **목표**: raw 텍스트 → sharded parquet → 다운로드/스트리밍/DDP 분배 → 토큰화 → 고정길이 (input,target) 패킹 → GPU 전송. 그리고 토큰 ID 한 개가 메모리에서 물리적으로 무엇인지 전자 수준까지.
- **읽을 코드**: `dataset.py`(전체), `dataloader.py`(전체), `repackage_data_reference.py`(데이터가 어떻게 준비됐는지).
- **핵심 개념**:
  - **sharded parquet + row group**(독립적으로 읽히는 ~1024 문서 블록) = 스트리밍·DDP striping의 단위 (`dataset.py:67-81`, `parquets_iter_batched`)
  - **DDP striping**: rank `r`이 `start=rank, step=world_size`로 row group을 읽어 코디네이터 없이 disjoint 분배 (`dataset.py:78`, `dataloader.py`의 `rg_idx += ddp_world_size`)
  - **BOS-aligned best-fit 패킹**: 가장 큰 doc를 통째로, 안 맞으면 가장 짧은 doc를 crop. 100% 활용(패딩 없음), ~35% 토큰 폐기 (`dataloader.py`)
  - **input/target shift**: `inputs=row[:,:-1]`, `targets=row[:,1:]` → `target[t]=input[t+1]` (다음 토큰 예측의 지도신호)
  - **legacy fallback 함정**: `base_data_climbmix`가 없으면 `dataset.py:38-58`이 *조용히* 옛 `base_data`로 redirect.
- **하드웨어 깊이**:
  - 1비트의 실체: 전압/전하로서의 0과 1, MOSFET = 게이트 전압으로 채널을 여닫는 스위치
  - DRAM 1T1C 셀(커패시터 전하=비트, 누설→refresh, 파괴적 읽기→sense amp) vs SRAM 6T 래치
  - `int64` 토큰 ID가 8바이트로 선형 주소공간에 놓이는 모습 (vocab=2¹⁵라 15비트면 되지만 인덱싱 때문에 int64)
  - **pinned(page-locked) host memory의 진짜 이득**: pageable도 DMA는 된다. pinned는 *OS가 페이지를 스왑아웃 못 하게 해 비동기 DMA에서 staging 복사를 생략*하고 copy/compute overlap을 가능케 한다 (`dataloader.py`의 `non_blocking` 복사).
- **직접 구현**:
  1. `python -m nanochat.dataset -n 3` → pyarrow로 한 파일 열어 `num_row_groups` 확인. **'text' 컬럼 한 문서를 출력해 "토큰 ID가 아니라 텍스트"임을 눈으로 확인**.
  2. ⚠️ **토크나이저 의존성 회피**: 데이터로더는 학습된 `tokenizer.pkl`을 요구한다(Stage 2 산출물). 아직 없으니 **`RustBPETokenizer.from_pretrained('gpt2')`를 fallback으로** 써서 `B=2, T=8` 한 배치를 뽑고 `torch.equal(inputs[:,1:], targets[:,:-1])`와 "모든 row가 BOS로 시작"을 assert.
  3. 패킹 루프를 계측해 T=2048에서 ~35% crop 폐기 재현. `T`/`buffer_size`를 스윕해 crop%가 어떻게 변하는지 차트.
- **자가시험**: 왜 `row_capacity`가 `T`가 아니라 `T+1`인가(off-by-one)? 토큰 ID 비트가 DRAM 커패시터에서 물리적으로 무엇이며 왜 refresh가 필요한가?
- **나중에 되짚을 것**: "BOS 정렬이 *왜* 더 나은가"(cross-document attention 오염)는 causal masking을 배우는 **Stage 3 직후 되짚는다**. 여기서는 "100% 활용 vs 폐기"의 *기계론*만.

---

### Stage 2 — 토크나이저 (1주)
- **목표**: Stage 1의 "토큰 ID"의 정체를 정의. 256 바이트 → BPE merge → vocab, GPT-4식 regex 분할, 특수토큰/BOS, vocab-불변 지표 bits-per-byte.
- **읽을 코드**: `tokenizer.py`(전체), `scripts/tok_train.py`, `scripts/tok_eval.py`.
- **핵심 개념**:
  - **바이트가 기본 알파벳**(`byte_fallback=True`): `unk_token` 없이 모든 입력을 무손실 표현. `vocab_size_no_special >= 256` 하한
  - **BPE merge**: 인접쌍 빈도 → 최빈쌍 병합 → 새 id. 학습은 `rustbpe`, 추론은 `tiktoken`으로 동일 재현
  - **GPT-4식 `SPLIT_PATTERN`**과 `\p{N}{1,2}`(숫자 2자리) 튜닝 — 32K vocab 예산에서의 trade-off
  - **특수토큰 9개**: BPE 밖에서 예약, vocab 상단에 배치. `encode("<|bos|>")`는 그 문자열을 *평범한 텍스트로* 토큰화함(진짜 삽입은 `prepend`/`encode_special`) — 고전적 함정
  - **bits-per-byte**: `bpb = total_nats / (log(2) × total_bytes)` → vocab이 달라도 비교 가능
- **하드웨어 깊이**: UTF-8 인코딩(코드포인트 → 1~4바이트, 한국어 같은 멀티바이트가 토큰 경계에서 쪼개져도 byte-level로 무손실 복원), 정수는 부동소수와 달리 exact, `wte(idx)`는 ID를 임베딩 테이블 주소로 쓰는 **포인터 산술**, 정보이론(엔트로피, nat vs bit, `log(2)`의 의미).
- **직접 구현**:
  1. `tok_eval.py`에 박혀 있는 `BasicTokenizer`(`get_stats`/`merge`)를 떼어내 완성. 한 문단에 `vocab_size=300` 학습하며 각 merge 출력 + `encode(decode(x))==x` 검증.
  2. `python -m scripts.tok_train --vocab-size 8192` 로 8K 토크나이저 직접 학습 → emoji/한국어/코드 round-trip assert.
  3. **(한국어 화자에게 특히 동기부여)** `tok_eval.py`로 도메인별 compression ratio 비교: 내 32K 토크나이저는 in-domain 영어에선 GPT-2를 이기지만 **한국어는 GPT-4(100K)에 진다**. byte-fallback과 vocab-size trade-off를 체감.
- **자가시험**: `bpb`가 왜 vocab-invariant이며 `log(2)`를 빼먹으면 무엇이 되나? 한국어에서 byte-level fallback이 무손실 decode를 보장하는 메커니즘은?
- **이관**: `render_conversation`의 supervision mask는 **Stage 9로 미룬다**. 여기선 특수토큰이 채팅 경계를 만든다는 한 줄 예고만.

---

### Stage 3 — GPT forward pass (2.5~3주) ⭐ 가장 깊은 단계
- **목표**: `gpt.py`를 처음부터 끝까지. embedding → RMSNorm → RoPE → multi-head attention(GQA/QK-norm/sliding window) → squared-ReLU MLP → pre-norm residual → lm_head → softcap → cross-entropy의 모든 텐서 shape를 추적. 동시에 `F.linear`가 결국 무엇인지 게이트/MAC/텐서코어까지.
- **읽을 코드**: `gpt.py`(전체 463줄).
- **핵심 개념** (✅ 현재 코드 직접 검증):
  - **custom `Linear`**(`gpt.py:45-50`): `F.linear(x, weight.to(x.dtype))` — fp32 master weight, bf16 matmul. 도크스트링이 *"Replaces autocast"*라고 명시. autocast를 일부러 안 쓴다.
  - **파라미터 없는 RMSNorm**(`gpt.py:42-43`), pre-norm Block `x = x + attn(norm(x)); x = x + mlp(norm(x))` (`gpt.py:146-149`)
  - **RoPE rotate-half**(`gpt.py:57-63`): `q·k`가 상대위치 `m-n`에만 의존. **연속 두 절반**(`x[...,:d]`, `x[...,d:]`) 컨벤션 — interleaved 아님(함정!)
  - **MHA shape 회계**(`gpt.py:82-124`): `(B,T,C)→(B,T,H,D)→(B,T,C)`. GQA(`n_kv_head` 공유), **QK-norm은 RoPE 후** 적용(`99-100`: `apply_rotary_emb` 다음 줄에 `norm(q), norm(k)`). `c_proj` zero-init이라 init 시 attention 출력 0.
  - **value residual(ResFormer)**: `gate = 2*sigmoid(ve_gate(x[..., :32]))` range (0,2), `v = v + gate*ve` (`gpt.py:92-95`). gate weight zero-init → init 시 gate=1.0 중립.
  - **lm_head untied + 작은 init**(std 0.001, `gpt.py:212`): 초기 loss ≈ `ln(vocab)`. logit softcap `20*tanh(logits/20)`(`gpt.py:419-423`), `cross_entropy(ignore_index=-1)`(`gpt.py:428`). logits는 softcap/loss 직전에 `.float()`로 fp32 복귀(`422`).
  - **resid/x0 lambda**: 매 레이어 `x = resid_lambdas[i]*x + x0_lambdas[i]*x0` (`gpt.py:413`), init resid=1.0 / x0=0.1 (`gpt.py:226-227`).
  - **meta device 함정**(`gpt.py:153-158`): `__init__`은 shape/dtype만(가짜 init). 실제 값은 `init_weights()`(`194-249`).
  - **vocab padding**: `pad_vocab_size_to=64`로 32768→올림(`gpt.py:166`), forward에서 다시 crop(`421`).
- **하드웨어 깊이** (여기가 핵심):
  - **IEEE-754 fp32(1+8+23) vs bf16(1+8+7)**: 같은 exponent → 같은 dynamic range, 줄어든 mantissa. 왜 master는 fp32(2⁻²³ epsilon이 작은 업데이트 보존), matmul은 bf16인가.
  - **`F.linear` = GEMM**: 출력원소 하나 = 길이 K dot product = K개 **MAC**(multiply-accumulate). 1 MAC = `estimate_flops`의 "2 FLOPs/weight"(`gpt.py:303`).
  - **MAC를 게이트로 짓기**: full adder(`sum=a⊕b⊕cin`, `carry=ab+cin(a⊕b)`), Wallace tree 곱셈기. bf16 곱셈 = 부호 XOR + 지수 정수가산 + 가수 정수곱 + 정규화/반올림.
  - **FMA**(한 번의 반올림으로 `a·b+acc`), **텐서코어** = systolic array*처럼* 동작하는 MAC 그리드가 16×16×16 MMA를 몇 사이클에(NVIDIA 마이크로아키텍처 비공개라 "처럼"), **SIMT warp** 32 lane이 lockstep.
  - vocab을 64배수로 padding, head_dim을 8배수로 맞추는 이유(GEMM 타일/MMA 정렬, `gpt.py:164-166`).
- **직접 구현**:
  1. tiny config(`n_layer=2, n_head=4, n_kv_head=2, n_embd=64, vocab=100`)로 `idx (2,16)` forward → 모든 중간 텐서 shape/dtype 출력. `loss ≈ ln(100)` 확인.
  2. `norm`, `apply_rotary_emb`를 직접 재구현. RoPE를 **interleaved로 바꿔 attention이 깨짐을 관찰**(왜 rotate-half인지 체득).
  3. fp32 워드 `0x42280000`을 손으로 디코드(=42.0), 상위 16비트만 남긴 bf16과 오차 계산. MLP/projection의 토큰당 MAC를 손으로 세어 `estimate_flops`의 `6*N + attention` 항과 대조.
- **자가시험**: `idx`→`logits`의 모든 shape를 추적할 수 있나? 두 `c_proj`가 zero-init이면 왜 init 시 모든 Block이 identity인가? bf16 곱셈이 게이트 수준에서 무엇을 하나(지수 가산 + 가수 곱)?

---

### Stage 4 — 정밀도와 커널 (1.5주)
- **목표**: Stage 3에서 black box였던 `flash_attn`과 dtype 캐스팅을 연다. no-autocast 전역 설계, fp8 scaling, flash attention의 tiling + online softmax.
- **읽을 코드**: `common.py`(COMPUTE_DTYPE 결정), `fp8.py`, `flash_attention.py`(전체).
- **핵심 개념** (✅ 현재 코드 직접 검증):
  - **전역 `COMPUTE_DTYPE`**(단일 import, no `torch.amp.autocast`) + fp32 master 불변식
  - **bf16 > fp16은 range 문제**(precision 아님): bf16은 fp32와 같은 8비트 exponent → `GradScaler` 불필요. fp16은 5비트 exponent → loss scaling 필요. `GradScaler`는 fp16 경로에서만. (`gpt.py:243-249` embeddings 캐스팅이 fp16만 예외 처리하는 이유)
  - **fp8 e4m3(입력/가중치) vs e5m2(그래디언트)**, tensorwise dynamic scaling(`scale=448/amax`)
  - **Flash Attention**(`flash_attention.py`): FA3 게이팅은 **SM90 Hopper + bf16만**(`flash_attention.py:28-61`), 그 외는 SDPA fallback. `_sdpa_attention`(`69-102`)이 sliding-window/cache 마스크를 명시적 boolean으로 구성. tiling으로 Q/K/V 블록을 SRAM에, online softmax(running max/sum 재스케일)로 T×T score를 HBM에 안 씀.
- **하드웨어 깊이**: fp8 비트 레이아웃(e4m3fn max 448, inf 없음), 메모리 계층(register/SRAM ~수십 TB/s → L2 → HBM 80GB ~3.3TB/s, 각 ~10×), **roofline & arithmetic intensity(FLOP/byte)**: flash attention이 O(T²) HBM 트래픽을 O(T)로 바꿔 bandwidth-bound를 compute-bound로 옮김.
- **직접 구현**: 5개 dtype의 비트수/max를 손계산 후 `torch.finfo`로 검증. `_to_fp8`을 직접 구현(amax→scale→clamp→cast). **online softmax를 직접 구현**해 full softmax와 일치 확인. `pytest tests/test_attention_fallback.py`(있으면).
- **선택(optional)**: `Float8Linear`/`_scaled_mm` row/col-major 레이아웃 심화는 **H100 전용**이라 CPU 학습자/캡스톤엔 불필요. 필수는 `COMPUTE_DTYPE` + bf16/fp16 range + flash tiling.
- **자가시험**: bf16과 fp16 차이가 precision이 아니라 range임을 한 문장으로? flash attention이 HBM 트래픽을 O(T²)→O(T)로 줄이는 핵심은?

---

### Stage 5 — 학습 루프와 옵티마이저 (2.5~3주) ⭐ 풀스택 종합
- **목표**: `loss=model(x,y) → backward → step → zero_grad` 한 스텝을 완전 해부. AdamW와 Muon(Newton-Schulz orthogonalization), LR/WD 스케줄, depth→모든 하이퍼파라미터 도출, DDP+ZeRO-2.
- **읽을 코드**: `scripts/base_train.py`(루프), `optim.py`(전체 534줄).
- **핵심 개념** (✅ 현재 코드 직접 검증):
  - **한 스텝 해부**: backward는 `.grad`를 *누적만*, `step()`이 `.data`를 in-place 변경, `zero_grad`로 리셋.
  - **gradient accumulation**: backward가 SUM이므로 `loss/grad_accum_steps`로 평균화.
  - **param grouping**(`gpt.py:356-394`, `setup_optimizer`): 2D transformer matrix → **Muon**(shape별 stacking, `383-388`), embeddings/lm_head/scalars → **AdamW**. AdamW LR을 `(dim/768)^-0.5`로 muP 스케일(`370`). 구체 LR: unembedding 0.004, embedding 0.2, matrix 0.02, scalar 0.5.
  - **Muon**(`optim.py:90-146`, `muon_step_fused`): Nesterov momentum(`109-112`) + Polar-Express 5-quintic Newton-Schulz(`82-88`, `114-127`)로 update의 특이값을 ~Uniform(0.5,1.5)로(근사 orthogonal). **tall/wide 행렬이 다른 분기**(`117` vs `122`). NorMuon 분산 감소(`129-140`) + cautious WD `mask=(g*params)>=0`(`145`).
  - **`MuonAdamW`(단일 GPU, `152`)** vs **`DistMuonAdamW`(`297`)**: 후자는 DDP all-reduce 없이 `step()` 안에서 `reduce_scatter`/`all_gather`로 ZeRO-2 샤딩(3-phase async overlap, `507-533`).
  - **`torch.compile` fused kernel**(`@torch.compile(dynamic=False, fullgraph=True)`, `20`·`90`) + **0-D CPU scalar tensor**(`180-192`)로 하이퍼파라미터 바뀌어도 recompile 회피.
- **하드웨어 깊이 — 여기가 사다리 종합**:
  - **풀스택 트레이스(한 스텝)**: `loss=model(x,y)` → CPython bytecode → C++ dispatcher / `torch.compile` FX graph → ATen(`aten::linear`→cuBLAS GEMM) → 비동기 CUDA stream launch → PTX→SASS(sm_90) → warp scheduler(SIMT 32 lane) → ALU FFMA / 텐서코어 HMMA → clocked flip-flop 파이프라인 → 논리 게이트 → CMOS 트랜지스터 → **전자/전압**.
  - backward는 같은 op tree를 역방향 재생해 **연산량 ~2배**(6 FLOPs/param = 2 fwd + 4 bwd). (연산량이 2배지 *커널 launch 수*가 정확히 2배는 아님.)
  - `train_loss.item()`/`synchronize()`가 강제 CPU-GPU sync point인 이유.
- **직접 구현**:
  1. tiny depth=4 CPU run에서 한 weight의 `.data`/`.grad`를 4지점(forward 직후/backward 후/step 후/zero_grad 후)에 print해 "backward는 grad만, step이 weight를" 확인.
  2. `adamw_step_fused`를 plain PyTorch로 재구현해 fused kernel과 대조.
  3. **256×256 행렬에 Newton-Schulz를 `ns_steps=0..5`로 돌리고 매 스텝 `svdvals`로 특이값이 ~1(0.5~1.5)로 이동함을 관찰** — Muon의 핵심을 눈으로.
  4. `--depth=20`의 model_dim/target_tokens(ratio 10.5)/batch를 손계산 후 스크립트 로그와 대조. `TORCH_LOGS=output_code`로 생성된 커널 구경.
- **자가시험**: backward와 step의 역할을 정확히 구분? `loss`를 왜 `grad_accum_steps`로 나누나? `model(x,y)` 한 줄에서 전자까지의 11계단을 **양방향으로** 막힘없이? `DistMuonAdamW`는 어디서 동기화하나(backward 아님 — `step()` 안)?

---

### Stage 6 — 체크포인트·리포트 (반나절)
- **목표**: 모델/옵티마이저/메타가 디스크에 어떻게 쓰이고 meta device로 재구성되는지(`build_model`). 이후 모든 하위 단계의 공통 진입점 `load_model`을 여는 짧은 다리.
- **읽을 코드**: `checkpoint_manager.py`, `report.py`.
- **핵심**: 포맷(`model_<step>.pt` rank0만 · `optim_<step>_rank<r>.pt` per-rank 샤딩), 탐색 휴리스틱(`find_largest_model`), `build_model`(meta device→`to_empty`→`init_weights`→`load_state_dict`), 전후방 호환 패칭, `'_orig_mod.'` 접두사 제거(torch.compile).
- **하드웨어 깊이**: meta device(파라미터를 두 번 할당 안 하는 메모리 절약), bf16→fp32 캐스팅(CPU/MPS는 bf16 compute 미지원), per-rank optim 파일 = Stage 5 ZeRO-2와 직결.
- **자가시험**: 8-GPU vs 1-GPU run 후 각각 어떤 파일이? 왜 model은 rank0만, optim은 모든 rank가 쓰나?

---

### Stage 7 — 베이스 평가 (1주)
- **목표**: Stage 2에서 맛본 bpb를 per-token grid 수준으로 해부. multiple-choice likelihood 스코어링, few-shot 렌더링, DCLM CORE의 random-baseline centering.
- **읽을 코드**: `loss_eval.py`, `core_eval.py`, `scripts/base_eval.py`.
- **핵심**: `loss_reduction='none'`((B,T) grid, `gpt.py:428`이 이를 지원), `token_bytes`(특수토큰=0바이트)로 마스킹, `find_common_length`로 continuation span 찾기(off-by-one `losses[si-1:ei-1]`), MC likelihood(continuation mean loss 최소), CORE centering `(acc-0.01·baseline)/(1-0.01·baseline)`.
- **직접 구현**: 같은 텍스트를 nanochat vs gpt2 토크나이저로 → 평균손실은 다르나 bpb는 같음을 보여 **vocab-invariance 증명**. ARC 한 문제를 4프롬프트 렌더→`find_common_length`→`argmin`이 gold와 일치 확인. `core_eval`의 `.mean()`을 `.sum()`으로 바꿔 **길이 편향(짧은 답 선호)** 측정.
- **자가시험**: bpb가 왜 바이트로 나누나? CORE centering이 왜 필요한가(4지선다 25% floor vs 생성 0% floor를 평균내려면)?

---

### Stage 8 — 추론 엔진 (1.5주)
- **목표**: KV-cache의 존재 이유와 성장, prefill vs decode, rotary offset, sampling, 캐시 클로닝 다중샘플, forced-token 도구사용 state machine.
- **읽을 코드**: `engine.py`, `execution.py`(sandbox), `scripts/chat_cli.py`. 비교용으로 `gpt.py:434-463`의 naive `generate`.
- **핵심**: 학습은 `(B,T)` 한 번 parallel, 추론은 마지막 위치 logits만 필요한 sequential 루프(naive는 O(T²) 재계산). KV cache 사전할당 + `cache_seqlens`, **rotary offset `T0=kv_cache.get_pos()`**(`gpt.py:404`), `advance(T)`는 마지막 레이어 후 1회만(`gpt.py:118-119`). 도구사용: `<|python_start|>~end` 버퍼링 → `use_calculator` → `[output_start..output_end]` forced 주입(`execution.py` sandbox). `render_for_completion`이 CLI/web/RL/eval 공통 진입점.
- **하드웨어 깊이 — roofline 재방문(가장 선명한 사례)**:
  - decode(`T_new=1`)는 토큰당 K/V를 `pos×H×D`개 read하는데 원소당 ~2 FLOP, 원소당 2바이트 read → **arithmetic intensity ≈ 1 FLOP/byte**. GPU break-even(H100 ≈ 990 TFLOPS / 3.35 TB/s ≈ **295**) 훨씬 아래 → SM이 메모리 대기로 idle = **bandwidth-bound**. prefill은 Q를 T번 재사용 → compute-bound.
  - KV cache footprint: `2 × n_layers × B × T × n_kv_head × head_dim × 2bytes`. d12, T=2048 = **72 MiB**. GQA가 캐시를 줄이는 이유.
- **직접 구현**: naive `model.generate` == cached `Engine.generate` 등가성 테스트(**temperature=0/argmax로** 검증). **`gpt.py:404`의 `T0`를 0으로 강제해 cached 경로가 garbage로 발산함을 관찰**. prefill vs per-token decode latency 분리 측정. **intensity≈1을 손계산으로 유도**.
- **자가시험**: 왜 decode는 bandwidth-bound이고 prefill은 compute-bound인가를 arithmetic intensity로? rotary offset `T0`를 빼먹으면 왜 깨지나?

---

### Stage 9 — SFT (1주)
- **목표**: 사전학습된 next-token 예측기를 채팅 어시스턴트로. **pretraining과 동일한 cross-entropy를 쓰되 assistant 토큰만 mask=1로 감독**.
- **읽을 코드**: `scripts/chat_sft.py`, `tasks/*`, `tokenizer.py`의 `render_conversation`.
- **핵심**:
  - 세 목표 구분: pretraining(모든 토큰) · **SFT(assistant 토큰만 마스킹된 동일 loss)** · RL(자기생성+reward). SFT는 pretraining loss에서 **target 마스킹만** 바뀜.
  - `render_conversation`이 반환하는 `(ids, mask)`: assistant text/python/`assistant_end`=1, BOS·user·`python_output`=0.
  - **`python_output`이 왜 mask=0**: 테스트 시 실제 인터프리터가 만드는 토큰이라, 감독하면 모델이 도구 결과를 *환각*하게 됨.
  - loss masking: `targets[mask==0] = -1` → `cross_entropy(ignore_index=-1)`(`gpt.py:428`과 연결). mask는 target과 함께 shift(`mask[:,1:]`).
  - `Task` 추상화 + `TaskMixture`(데이터 믹스가 곧 SFT 커리큘럼), SFT dataloader는 **pad-not-crop**(절대 crop 안 함).
  - base→sft 계보: `load_model('base')`, LR 리셋, `weight_decay=0`.
- **직접 구현**: gsm8k/spellingbee conversation 하나를 `visualize_tokenization(ids, mask)`로 색칠해 빨강(mask 0)/초록(mask 1) 눈으로 확인. 간단 능력 `Task` 서브클래스 작성해 mixture에 추가.
- **자가시험**: SFT가 pretraining loss에서 정확히 무엇만 바꾸나? `python_output`을 mask=1로 하면 어떤 버그가? mask를 함께 shift 안 하면?

---

### Stage 10 — RL(GRPO) + 채팅 평가 (1.5주)
- **목표**: 모방을 넘어 모델 자신의 생성을 reward로 강화. rollout, group-relative advantage, token-level 정책경사. + 채팅 평가로 파이프라인 종료.
- **읽을 코드**: `scripts/chat_rl.py`, `scripts/chat_eval.py`.
- **핵심**: rollout(`engine.generate_batch`, per-step seed 변경, `train_task.reward`로 0/1 채점), **단순화 GRPO advantage = `rewards - rewards.mean()`**(std 나눗셈 없음·critic 없음), 정책경사 `pg_obj=(logp*adv).sum()/num_valid`(`logp = -model(loss_reduction='none')`), on-policy라 PPO clipping 불필요. 평가: categorical(letter-logit argmax) + generative pass@k + ChatCORE.
- **직접 구현**: advantage가 모두 0이 되는 질문 비율 로깅(pass@k와 연결 — 너무 쉽거나 어려운 문제는 gradient 0). **logp의 마이너스 부호를 제거해 gradient 방향이 뒤집힘(reward 하락)을 관찰**.
- **자가시험**: SFT(imitation)와 RL(reward 최적화)의 목표가 어떻게 다른가? group baseline이 critic 없이 어떻게 variance를 줄이나? per-sample seed를 고정하면 왜 gradient가 0이 되나?

---

## 5. ★ 캡스톤 — 무참조 재구현 (1.5~2주)

원본을 **보지 않고** (개념 + 자가 작성 노트만으로) 미니 GPT를 데이터→토크나이저→학습→추론까지 처음부터 작성한다. `depth=4, n_embd=128`, 초소형 코퍼스(셰익스피어/위키 한 셰어드), **GPU 없으면 CPU(fp32)로 전부** — 이것이 nanochat이 단일 노드로 설계된 이유를 체감하는 핵심.

**재구현 순서** (각 모듈 완성 시마다 원본과 대조, 막힌 줄만 열어보고 다시 가림):

1. **byte-level BPE 토크나이저**: 256 base, `get_stats`/`merge` 학습, `<|bos|>` 특수토큰. → emoji/한국어/코드 `encode(decode(x))==x`
2. **데이터로더**: 토큰화→BOS prepend→best-fit 패킹→`row_capacity=T+1`로 input/target shift. 무한 generator.
3. **기반 모듈**: `GPTConfig`, custom `Linear`(bias 없음, `weight.to(x.dtype)`), 파라미터 없는 RMSNorm, RoPE(rotate-half), QK-norm(RoPE 후).
4. **attention+MLP+block**: `CausalSelfAttention`(GQA/causal, `c_proj` zero-init), squared-ReLU MLP, pre-norm Block.
5. **`GPT.forward` 조립**: wte→norm→블록 루프→final norm→lm_head(untied, tiny init)→fp32 softcap→cross_entropy. **init loss ≈ `ln(vocab)` 확인.**
6. **학습 루프**: forward→loss/grad_accum→backward→step→zero_grad, LR warmdown. **먼저 표준 AdamW로 loss 하락 확인**, 그다음 Muon 추가. ⚠️ Polar-Express 계수·tall/wide 분기·NorMuon은 암기 불가 → **Stage 5 노트/논문 참조 허용**, 단 "Nesterov momentum + Newton-Schulz" 골격은 직접.
7. **평가**: `token_bytes` + bits-per-byte.
8. **추론**: `KVCache` + `Engine.generate`(prefill→single-token decode, rotary offset, sampling). **cached == naive 토큰 단위 등가성**(temperature=0/argmax로 검증), `T0=0`으로 강제 시 깨짐 재현.
9. (선택) **SFT**: assistant-only mask로 작은 합성 채팅 Task 미세조정.

**성공 기준**:
- [ ] 토크나이저: 모든 도메인 무손실 round-trip
- [ ] 데이터로더: 모든 row가 BOS로 시작, `torch.equal(inputs[:,1:], targets[:,:-1])`
- [ ] 모델: forward shape가 손계산과 일치, init loss가 `ln(vocab)`의 ~1% 이내
- [ ] 학습: 미니 코퍼스에서 loss 단조 하락, Muon vs AdamW 수렴 비교(*반드시* 우세일 필요는 없음)
- [ ] 추론: cached==naive(argmax), `T0=0` 강제 시 garbage 재현
- [ ] bpb: 동일 고정 텍스트를 두 토크나이저(8K/16K)로 토큰화 후 같은 작은 모델로 per-token loss vs bpb 비교(학습 1회만)
- [ ] **하드웨어**: 자신이 짠 `F.linear` 한 줄을 `CPython→ATen GEMM→텐서코어 MAC→게이트→트랜지스터→전자`로, 그리고 **역방향으로** 막힘없이. 자신의 `KVCache`가 왜 decode를 bandwidth-bound로 만드는지 roofline으로.

---

## 6. 함정 체크리스트 (자가점검용)

이것만 다 설명할 수 있으면 코어를 이해한 것이다:

- [ ] **`row_capacity = T+1`** (T 아님): 이 +1이 있어야 input/target이 둘 다 길이 T. next-token 예측의 가장 고전적 off-by-one.
- [ ] **셰어드는 토큰이 아니라 텍스트를 저장**: ClimbMix 원본은 GPT-2 토큰이라 repackage가 decode→텍스트로 되돌려 저장, 로드 시 nanochat 토크나이저로 재인코딩.
- [ ] **`encode("<|bos|>")`는 특수토큰을 안 만든다** — 그 문자열을 평범한 텍스트로 토큰화. 진짜 삽입은 `prepend`/`encode_special`.
- [ ] **RoPE는 rotate-half(연속 두 절반)지 interleaved 아님**. q/k가 같은 컨벤션 안 쓰면 attention이 *조용히* 망가짐. QK-norm은 RoPE **후**(`gpt.py:99-100`).
- [ ] **lm_head는 wte와 untied**(std 0.001 vs 1.0). weight tying 가정 금지. 두 `c_proj`는 zero-init이라 init 시 모든 Block이 identity(버그 아님, 의도).
- [ ] **`__init__`은 meta device**(shape/dtype만). 실제 값은 `init_weights()`.
- [ ] **bf16 vs fp16은 range 문제지 precision 아님**. `GradScaler`는 fp16에서만.
- [ ] **autocast가 일부러 없다**(`gpt.py:45-50` 도크스트링). `Linear`가 forward마다 fp32→`x.dtype` 캐스팅, embeddings 사전캐스팅(`243-249`), logits만 fp32 복귀(`422`).
- [ ] **fp8 캐스팅 전 clamp 필수**(PyTorch fp8 캐스트는 saturate 아니라 wrap).
- [ ] **loss를 `grad_accum_steps`로 나누는 이유**: backward가 SUM 누적. backward는 `.grad`만, step이 `.data`.
- [ ] **Muon은 2D matrix에만**. orthogonalization은 일부러 완전수렴 안 하고 특이값을 ~Uniform(0.5,1.5)로(`optim.py:60-63` 주석).
- [ ] **`DistMuonAdamW`는 DDP로 안 감싼다** — backward에 all-reduce 없음. 동기화는 `step()` 안에서 reduce_scatter/all_gather(`optim.py:507-533`).
- [ ] **decode는 bandwidth-bound, prefill/학습은 compute-bound.** rotary offset `T0`(`gpt.py:404`)를 빼먹으면 모든 생성 토큰이 position 0 위상 → garbage. `advance(T)`는 마지막 레이어 후 1회만.
- [ ] **SFT loss mask는 target과 함께 shift**(`mask[:,1:]`). `python_output`은 SFT/RL 양쪽 mask=0.
- [ ] **RL: per-sampling-step마다 seed 변경.** 고정 seed → 동일 샘플 → reward 분산 0 → advantage/gradient 0. `logp`는 모델 loss의 **음수**.
- [ ] **bpb는 바이트로 나누고 `log(2)` 포함**. 빼먹으면 nats-per-byte.
- [ ] **legacy fallback 함정**: `base_data_climbmix`가 없으면 `dataset.py:38-58`이 *조용히* 옛 `base_data`로 redirect → 절반만 마이그레이션하면 엉뚱한 데이터로 학습.

---

## 7. 진행 추적

```
[ ] Stage 0  오리엔테이션
[ ] Stage 1  데이터 파이프라인
[ ] Stage 2  토크나이저
[ ] Stage 3  GPT forward          ⭐
[ ] Stage 4  정밀도·커널
[ ] Stage 5  학습 루프·옵티마이저   ⭐
[ ] Stage 6  체크포인트·리포트
[ ] Stage 7  베이스 평가
[ ] Stage 8  추론 엔진
[ ] Stage 9  SFT
[ ] Stage 10 RL + 채팅 평가
[ ] ★ 캡스톤 무참조 재구현
```

---

## 부록 A — 핵심 파일 참조 지도

| 파일 | 역할 | 가장 중요한 줄/심볼 |
|---|---|---|
| `nanochat/gpt.py` | 트랜지스터의 심장 | `Linear`(45), `apply_rotary_emb`(57), `CausalSelfAttention`(65), QK-norm(99-100), `MLP`(127), `init_weights`(194), `setup_optimizer`(356), `forward`(396), softcap(419-423), naive `generate`(434) |
| `nanochat/optim.py` | AdamW + Muon | `adamw_step_fused`(20), `polar_express_coeffs`(82), `muon_step_fused`(90), tall/wide(117-126), cautious WD(145), `MuonAdamW`(152), `DistMuonAdamW`(297), 3-phase `step`(507) |
| `nanochat/dataset.py` | parquet 셰어드 | `list_parquet_files`(32, legacy fallback 38-58), `parquets_iter_batched`(67), `download_single_file`(84) |
| `nanochat/dataloader.py` | best-fit 패킹 | BOS 정렬, `row_capacity=T+1`, input/target shift, pinned `non_blocking` |
| `nanochat/tokenizer.py` | BPE | `SPLIT_PATTERN`, `RustBPETokenizer`, `render_conversation`(ids,mask), `get_token_bytes`, `visualize_tokenization` |
| `nanochat/flash_attention.py` | FA3/SDPA | FA3 게이팅(28-61), `_sdpa_attention`(69), `flash_attn_func`(107), `flash_attn_with_kvcache`(131) |
| `nanochat/common.py` | 전역 설정 | `COMPUTE_DTYPE`, `get_dist_info`, `get_base_dir` |
| `nanochat/engine.py` | KV-cache 추론 | `KVCache`, `Engine.generate`, prefill/decode |
| `nanochat/checkpoint_manager.py` | 저장/복원 | `build_model`, `load_model`, `find_largest_model` |
| `runs/speedrun.sh` | 정식 파이프라인 | 전체 실행 순서의 정답지 |

---

*작성: 2026-06-30. nanochat 리포(`/home/sgkim/nanochat`, branch `dev`) 전체 분석 + 핵심 파일 직접 정독 검증 기반.*
