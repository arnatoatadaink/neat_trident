# 判断記録: PyTorch本格移植の可否 (pytorch-full-port-decision)

日付: 2026-07-14
関連タスク: `pytorch-full-port-decision`
関連文書: `docs/handoff_pytorch_rocm_prototype_20260713.md`,
`docs/rocm_pcie_passthrough_eval_20260714.md`

## 結論: 見送り (JAX-ROCm検証をゲートとして先行させる)

PyTorchへの本格移植は **現時点では着手しない**。代わりに `jax-rocm-spike`
（JAX-ROCmが実機で動くかの検証）を先に行い、その結果次第で本判断を再訪する。

## 判断根拠

### 1. JAX-ROCmという「未検証の前提」が閉じられないまま進んでいた

`docs/handoff_pytorch_rocm_prototype_20260713.md` の背景にある通り、そもそも
PyTorchプロトタイプ着手の理由は「AMD GPU導入予定 (JAX-ROCmは未検証)」だった。
JAX-ROCmは「否定された」のではなく「検証されずスキップされた」だけであり、
本格移植のコストをかける前に、まずこの未決の前提を閉じるべき。

JAX-ROCmが動けば:
- 既存の `src/es_hyperneat.py`(518行)・`src/map_elites_archive.py`(435行、QDax依存)・
  `src/novelty_search.py`(470行)・`src/interfaces/*`(3ファイル計約1268行)・
  `src/med_integration/*` 一式 — これらをほぼ無改修のままAMD GPUで動かせる可能性がある。
- PyTorchへの本格移植（種分化・交叉・世代ループの新規実装 + 上記全ファイルの
  TensorNEAT依存の置き換え、約4000行規模）というコストの大部分が不要になる。

つまりJAX-ROCmの検証コストは小さく、当たれば最大のコスト（本格移植）を丸ごと回避できる
非対称な賭けであり、先にここを潰すのが合理的。

### 2. ROCmベンチ結果は「本格移植の十分条件」を示していない

`docs/rocm_pcie_passthrough_eval_20260714.md` の実測:

| label | CPU mean_forward_ms | ROCm(cuda) mean_forward_ms | 速度向上 |
|---|---|---|---|
| Spiral (pop=100, samples=200) | 1268.083 | 31.303 | 約40.5倍 |
| VectorNeighbor (pop=100, samples=20) | 67.65 | 21.069 | 約3.2倍 |

これは **順伝播のみ** の計測であり、種分化・交叉・選択を含む世代ループ全体の
計測ではない（プロトタイプ自体に世代ループが未実装）。

一方、既存のJAX実装は **RTX 4060上で世代ループ全体を含めて** 以下の速度向上を
既に達成済み（`memory: project_gpu_env.md`）:

| 問題 | CPU | RTX4060(JAX) | 速度向上 |
|---|---|---|---|
| Spiral | 238ms/gen | 14ms/gen | 17x |
| VectorNeighbor | 188ms/gen | 22ms/gen | 8.5x |

「ROCmで40倍速い」という数字と「JAXは17倍」という数字は測定対象が異なり
（順伝播のみ vs 全体ループ）単純比較できない。本格移植によって得られる
**限界的な性能向上は現時点で未証明** — 少なくとも順伝播単体の40倍がそのまま
世代ループ全体に乗る保証はない。

### 3. WSL2固有のクラッシュ経路はJAX-ROCmでも再発しうる

PyTorch側で発生した `rocprofiler-sdk` のクラッシュ（HSAエージェント数とKFD
topology sysfs — WSL2には存在しない — の不整合による`FATAL abort`）は、
`ROCPROFILER_REGISTER_ENABLED=0` で回避できることが判明済み。しかし、これは
PyTorch同梱コンポーネント固有の問題であり、JAX-ROCmが同種のWSL2起因の問題
（`/dev/dxg` 経由の準仮想化GPU、ネイティブamdgpuドライバ非ロード）を持たない
保証はない。JAX-ROCm自体のWSL2上での動作実績は未確認。

またgfx1201(RX 9070 XT)/gfx1200(RX 9060 XT)はRDNA4世代でごく新しく、
jaxlibのROCmビルドがこのアーキテクチャ・ROCm 7.x系列をサポートしているかも
未確認 — ここもjax-rocm-spikeで確認すべき点。

## 移植した場合のスコープ（参考・現時点では未着手）

仮に本格移植に進む場合、対象は以下（合計約4000行規模、種分化/交叉/世代ループの
新規実装を含む）:

- `src/es_hyperneat.py`, `src/map_elites_archive.py`(QDax依存 → 代替ライブラリ選定要),
  `src/novelty_search.py`
- `src/interfaces/{neat_gate,neat_indexer,neat_slot_filler}.py`
- `src/med_integration/*`(NEATAssociationFn等、ゲノム表現に依存する箇所)
- `src/pytorch_proto/` 自体への種分化・交叉・世代ループの追加実装

## 次のアクション

1. `jax-rocm-spike`（新規登録, P3）— jax-rocm-plugin（またはjaxlib ROCmビルド）を
   実機（RX 9070 XT / RX 9060 XT, WSL2）にインストールし、`jax.devices()`が
   GPUを認識するか、簡単なNEAT世代ループが動くかを確認する。
   `ROCPROFILER_REGISTER_ENABLED=0` 等の環境変数が同様に必要になる可能性を含め検証。
2. `jax-rocm-spike` の結果を踏まえ、`pytorch-full-port-decision` を再訪:
   - JAX-ROCmが動く → 本格移植は不要、既存JAXコードをそのままAMD機で使う方針に転換
   - JAX-ROCmが動かない → 改めてPyTorch本格移植のコスト対効果を判断
     （その際は世代ループ込みのROCmベンチを取り直すこと）

## 未回答の前提

なぜRTX 4060(JAX/CUDA)が既に動いている状況でAMD GPU導入を検討しているのか
（VRAM増強、複数GPU構成での並列化など）は本セッションでは確認していない。
動機によって本判断の優先度が変わりうるため、次回この判断を再訪する際は
ユーザーに確認することを推奨する。
