# Stage 1 — 데이터 파이프라인 학습 설계

> `LEARNING_PLAN.md`의 Stage 1을 **실전 학습용으로 확장**한 문서.
>
> **한 줄 목표**: raw 텍스트 한 덩어리가 어떻게 `(B, T)` 정수 텐서 배치로 바뀌어 GPU에 도착하는지, 그 여정의 **모든 관문**을 이해한다. 그리고 그 정수 한 개가 메모리에서 물리적으로 무엇인지 전자 수준까지.
>
> **학습 원칙**: "동작을 먼저 보고 → 데이터가 흐르는 방향을 따라가며 → 바닥까지 판다."
>
> *line 인용은 2026-07-14 `dev` 브랜치 코드 직접 정독 검증 기준. `dataset.py`(161줄)·`dataloader.py`(167줄) 전체 확인.*

---

## 데이터의 여정 = 학습 순서

이 화살표 하나하나가 학습 단계다. **데이터가 흐르는 순서 그대로** 배운다.

```
 parquet 셰어드        DDP 분배         토큰화          best-fit 패킹      shift        GPU
 (텍스트 문서)   →   (rank별 분배)  →  (BOS 붙여    →   (T+1 길이 행    →  input/    →  (B,T)
                                       정수 리스트)      로 꽉 채움)        target       텐서
   §1.1              §1.3            §1.4-a           §1.4-b           §1.4-c       §1.5
   dataset.py        dataloader.py   dataloader.py    dataloader.py                dataloader.py
                                                                                      │
                                            §1.6 ← 이 정수 1개가 물리적으로 무엇인가 (전자·DRAM)
```

## 학습 순서 개요

| 순서 | 무엇을 | 읽을 코드 (검증된 줄) | 시간 |
|---|---|---|---|
| **1.0** | **끝그림 관찰** (코드 읽기 전, 눈으로) | 실행만 | 30분 |
| **1.1** | 저장 계층: 셰어드 parquet | `dataset.py` (전체 161줄) | 반나절 |
| **1.2** | 데이터의 출처 (선택) | `dev/repackage_data_reference.py` | 30분 |
| **1.3** | DDP 분배: striping | `dataloader.py:25-71` (`_document_batches`) | 반나절 |
| **1.4** ⭐ | **핵심: best-fit BOS 패킹 + shift** | `dataloader.py:74-161` | 1.5일 |
| **1.5** | GPU 전송: pinned memory | `dataloader.py:111-160` | 반나절 |
| **1.6** | 하드웨어 바닥: 비트의 실체 | (코드 아님, 개념) | 1일 |

> **⚠️ 어디서 실행하나**: 코드가 있는 곳은 `dev` worktree(`/home/sgkim/nanochat`)다. `study` worktree는 main 파생이라 코드가 없다(README + 학습문서만). **실습은 `/home/sgkim/nanochat`에서** 돌리고, 노트/직접 짠 실험 스크립트만 study에 정리한다.

---

## §1.0 — 끝그림 관찰 (코드 읽기 전에!)

**목표**: 입력(텍스트)과 출력(`(B,T)` 정수 텐서)을 **눈으로** 먼저 본다. 아직 어떻게 되는지는 몰라도 된다.

**할 것** (토크나이저 없이도 되는 것부터):
```bash
cd /home/sgkim/nanochat && source .venv/bin/activate
python -m nanochat.dataset -n 1          # 셰어드 1개 + val 셰어드 다운로드
```
그다음 파이썬에서 **raw 텍스트가 저장돼 있음**을 직접 확인 (토크나이저 불필요):
```python
from nanochat.dataset import parquets_iter_batched
texts = next(parquets_iter_batched("train"))   # 첫 row group의 문서들
print(type(texts), len(texts))                 # list, ~1024개
print(texts[0][:500])                          # ← "토큰 ID가 아니라 사람이 읽는 텍스트!"
```

**이해 체크**: 셰어드 안에 저장된 게 토큰인가 텍스트인가? (답: **텍스트**. 이게 §1.2의 함정과 직결)

---

## §1.1 — 저장 계층: 셰어드 parquet (`dataset.py`)

**목표**: 데이터가 디스크에 *쉬고 있을 때*의 형태를 이해.

**핵심 개념** (✅ 현재 코드 검증):
- **6542개 셰어드**(`MAX_SHARD=6542`), 파일명 `shard_00000.parquet`, HF ClimbMix-400B에서 on-demand 다운로드(`dataset.py:23-27`).
- **row group** = parquet 내부의 독립적으로 읽히는 문서 블록(~1024개). 스트리밍·DDP 분배의 **단위**(`parquets_iter_batched:67-81`).
- **train/val split**: `paths[:-1]`이 train, **마지막 셰어드 하나가 val**. 다운로드 시 val 셰어드(index 6542)는 `-n`과 무관하게 **항상** 받음(`dataset.py:146-149`).
- **⚠️ legacy fallback 함정**(`dataset.py:38-58`): `base_data_climbmix/`가 없으면 *조용히* 옛 `base_data/`로 redirect. `warn_on_legacy=True`일 때만 경고. → 데이터 절반만 마이그레이션하면 엉뚱한 데이터로 학습.

**이해 체크**: `-n 8`을 주면 실제로 몇 개 파일이 받아지나? (답: 8 + val 1 = **9개**)

---

## §1.2 — 데이터의 출처 (선택, 30분)

**목표**: "셰어드는 왜 텍스트를 저장하나"의 답. `dev/repackage_data_reference.py`를 훑어봄.

**핵심**: ClimbMix 원본은 **GPT-2 토큰**으로 배포됨. repackage가 이걸 `decode` → 텍스트로 되돌려 parquet에 저장. 그래야 로드할 때 **우리 nanochat 토크나이저로 재인코딩**할 수 있음. (남의 토큰에 묶이지 않으려는 설계)

---

## §1.3 — DDP 분배: striping (`dataloader.py:25-71`)

**목표**: GPU 8장이 **코디네이터 없이** 서로 겹치지 않는 데이터를 읽는 법.

**핵심 개념** (✅ 검증):
- rank `r`은 row group을 `rg_idx = ddp_rank`(`:62`)에서 시작해 `rg_idx += ddp_world_size`(`:68`)로 건너뜀 → 8장이면 rank0은 0,8,16… rank1은 1,9,17… **disjoint 자동 분배** (통신 0).
- **무한 반복**(`while True`, `:47`) + epoch 카운터 → 데이터 끝나면 다시 순환.
- resume 로직(`:53-60`)은 지금은 훑어만 봄(체크포인트 재개용).

**이해 체크**: 왜 이 striping은 GPU 간 통신이 전혀 필요 없나? (답: 각 rank가 자기 몫을 **산술로** 계산 — `start=rank, step=world_size`)

---

## §1.4 — ⭐ 핵심: best-fit BOS 패킹 (`dataloader.py:74-161`)

**여기가 Stage 1의 심장이다.** 1.5일 투자.

**목표**: 가변 길이 문서들을 낭비 없이 고정 길이 `T` 행에 채우고, next-token 지도신호를 만드는 법.

**핵심 개념** (✅ 검증):

**(a) 토큰화 + BOS**: `tokenizer.encode(doc_batch, prepend=bos_token, ...)`(`:100,107`) — encode 시점에 **BOS를 앞에 붙임**. 그래서 모든 문서가 BOS로 시작.

**(b) best-fit 패킹** (`:122-151`) — 한 행(`row_capacity = T+1`, `:98`)을 채우는 알고리즘:
```
1. 버퍼에서 "남은 공간에 통째로 들어가는 가장 큰 문서"를 골라 넣는다  (:135-145)
2. 더 이상 들어가는 문서가 없으면 → "가장 짧은 문서를 remaining만큼 잘라" 딱 채운다  (:148-150)
   → 100% 활용(패딩 0), 대신 T=2048에서 ~35% 토큰이 crop으로 버려짐
```

**(c) input/target shift** (`:154-155`): 한 행은 길이 `T+1`. 여기서
```python
inputs  = row[:, :-1]   # 길이 T
targets = row[:, 1:]    # 길이 T   →  targets[t] == inputs[t+1]  (다음 토큰 예측!)
```

**🔑 가장 중요한 통찰 — `row_capacity = T+1`인 이유**: input과 target이 **둘 다 길이 T**가 되려면 원본이 T+1이어야 한다. 이 `+1`이 next-token 예측의 **가장 고전적인 off-by-one**. 이거 하나가 Stage 1의 핵심 시험문제다.

**이해 체크**:
1. `T+1`이 아니라 `T`로 하면 무슨 일이? (input 또는 target이 T-1이 됨)
2. best-fit이 "가장 큰 것 먼저"인 이유는? (crop 횟수를 줄여 낭비 최소화)
3. crop된 35%는 왜 감수하나? → **나중에 되짚음**: BOS 정렬이 cross-document attention 오염을 막기 때문. 이 "왜 더 나은가"는 **causal masking을 배우는 Stage 3 직후 재방문**. 지금은 *기계론*만.

---

## §1.5 — GPU 전송: pinned memory (`dataloader.py:111-160`)

**목표**: CPU에서 만든 텐서가 GPU로 가는 **최적화된 복사 경로**.

**핵심 개념** (✅ 검증) — 버퍼 3단:
```
row_buffer (B, T+1)  ──shift──▶  cpu_buffer (2·B·T, pinned)  ──HtoD 1회──▶  gpu_buffer
행을 조립               :154-155      pin_memory=True (:115)      non_blocking (:160)
```
- **레이아웃 트릭**: `[inputs | targets]`를 한 평평한 버퍼에 이어 붙여 **단 한 번의 HtoD 복사**(`:159-160`).
- **pinned(page-locked) memory의 진짜 이득**: pageable 메모리도 DMA는 됨. pinned는 *OS가 그 페이지를 스왑아웃 못 하게 고정*해서, 비동기 DMA가 staging 복사를 생략하고 **copy/compute overlap**을 가능케 함(`non_blocking=True`가 의미를 가지는 전제).

**이해 체크**: `non_blocking=True`가 pinned 메모리 없이는 왜 무의미한가?

---

## §1.6 — 하드웨어 바닥: 토큰 ID 한 개의 물리적 실체 (1일)

**목표**: 방금 본 `torch.long` 토큰 버퍼의 정수 하나가 실리콘에서 **0과 1, 전자**로 무엇인지.

**채울 사다리 칸** (1~7단):
- **1단 전자/전압**: 비트 = 전하/전압 레벨. 0과 1은 노이즈 마진을 둔 두 전압 구간.
- **2단 MOSFET**: 게이트 전압으로 채널을 여닫는 스위치. 트랜지스터 = "전기로 제어되는 밸브".
- **6단 메모리 셀**: **DRAM 1T1C**(커패시터 전하 = 비트, 누설되니 **refresh** 필요, 읽으면 파괴돼 **sense amp**로 복원) vs **SRAM 6T** 래치(빠르지만 6배 크다).
- **7단 주소공간**: `int64` 토큰 ID 8바이트가 선형 주소공간에 놓임. vocab=2¹⁵이라 **15비트면 충분한데 왜 int64인가?** → 인덱싱·정렬·커널 호환. `pin_memory`가 이 페이지를 물리 주소에 **고정**한다는 것의 의미.

**이해 체크**: 토큰 ID 비트가 DRAM 커패시터에서 물리적으로 무엇이며, 왜 refresh가 필요한가?

---

## 직접 구현 — 실험 3개 (이해를 손으로 검증)

1. **row group 계측**: `pyarrow`로 셰어드 한 개 열어 `pf.num_row_groups` 확인, `'text'` 컬럼 한 문서 출력 → "텍스트지 토큰 아님" 눈으로.
2. **shift 불변식 assert** (⚠️ 여기서 토크나이저 필요 → `RustBPETokenizer.from_pretrained('gpt2')` fallback 사용. 우리 `tokenizer.pkl`은 Stage 2 산출물이라 아직 없음):
   ```python
   # B=2, T=8로 한 배치 뽑아서
   assert torch.equal(inputs[:, 1:], targets[:, :-1])   # target[t]=input[t+1]
   assert (inputs[:, 0] == bos_token).all()             # 모든 행이 BOS로 시작
   ```
   (실행 시 gpt2 fallback이 `get_bos_token_id()`를 제공하는지 함께 확인 — 안 되면 Stage 2를 미리 조금 당겨 8K 토크나이저를 먼저 학습)
3. **crop% 재현**: 패킹 루프를 계측해 `T=2048`에서 ~35% crop 폐기 재현. `T`/`buffer_size`를 스윕해 crop%가 어떻게 변하는지 표로.

---

## 자가시험 (이걸 다 답하면 Stage 1 통과)

- [ ] 왜 `row_capacity`가 `T`가 아니라 `T+1`인가?
- [ ] DDP striping이 통신 없이 disjoint 분배를 이루는 산술은?
- [ ] best-fit이 100% 활용을 얻는 대가로 무엇을 버리나? 그게 왜 괜찮나?
- [ ] pinned memory가 없으면 `non_blocking`이 왜 무의미한가?
- [ ] 토큰 ID 비트가 DRAM 셀에서 물리적으로 무엇이며 refresh가 왜 필요한가?
- [ ] legacy fallback 함정은 어떻게 조용히 사고를 내나?

## 1주 시간 배분 (주 10~15시간 기준)

```
1일차: §1.0 관찰 + §1.1 dataset.py 정독
2일차: §1.2(선택) + §1.3 DDP striping
3~4일차: §1.4 best-fit 패킹 ⭐ (실험 2번 포함)
5일차: §1.5 pinned memory + 실험 3번
6~7일차: §1.6 하드웨어 바닥 (전자·MOSFET·DRAM)
```

---

*작성: 2026-07-14. `dataset.py`·`dataloader.py` 전체 직접 정독 검증 기반. 상위 커리큘럼은 [`LEARNING_PLAN.md`](LEARNING_PLAN.md)의 Stage 1 참조.*
