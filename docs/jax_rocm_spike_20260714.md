# 検証記録: JAX-ROCm実機動作検証 (jax-rocm-spike)

日付: 2026-07-14
関連タスク: `jax-rocm-spike`
関連文書: `docs/pytorch_full_port_decision_20260714.md`,
`docs/handoff_pytorch_rocm_prototype_20260713.md`,
`docs/rocm_pcie_passthrough_eval_20260714.md`

## 結論: JAX-ROCmは現状このマシンでは実用不可

`pytorch_full_port_decision_20260714.md` で「未検証の前提」として残っていた
JAX-ROCmの実機動作を検証した。結果、**GPU認識の段階で重大な問題があり、
仮に回避してもNEAT世代ループが2世代目でクラッシュする**ことを確認した。
PyTorch-ROCmは既に順伝播ベンチマークまで動作確認済み(`rocm_pcie_passthrough_eval_20260714.md`)
であるのに対し、JAX-ROCmはこの時点で実用に耐えない。

## 環境

- GPU: AMD Radeon RX 9070 XT (gfx1201) + RX 9060 XT (gfx1200)、WSL2上
- ROCm: 7.2.3 (システム側、`rocminfo`は正常動作)
- 追加インストール: `jax-rocm7-pjrt==0.9.2`, `jax-rocm7-plugin==0.9.2`
  (`pip install`、poetry管理外・venv内のみ。既存の `jax==0.9.2`/`jaxlib==0.9.2` と
  バージョン一致させて選定)

## 検証手順と結果

### 1. `ROCPROFILER_REGISTER_ENABLED=0` は引き続き必要

PyTorchで判明していたのと同じ `rocprofiler-sdk` のクラッシュ経路が、環境変数
未設定だとJAX側でも発生した(`jax_plugins.xla_rocm7.initialize()` 内で
`agent.cpp:1093` の `FATAL abort`、HSAエージェント数とKFDトポロジーsysfsの
不整合)。`ROCPROFILER_REGISTER_ENABLED=0` を設定すると警告ログのみになり
先に進む。

### 2. 2GPU同時可視だと `jax.devices()` が45分以上ハング

`ROCR_VISIBLE_DEVICES` 未設定(gfx1200・gfx1201の両方が見える状態)だと、
`jax.devices()` 呼び出しが `PJRT_Client_Create`(`ROCmPlatform::VisibleDeviceCount()`)
内でCPU 290%・66スレッドのまま45分以上応答しない。py-spyでスタック確認済み
(ネイティブコール内で停止、Pythonフレームは`make_c_api_client`から進まず)。
クラッシュではなく、進捗の見えないハング。

参考: ROCm本体側にもRDNA4デュアルGPU構成でのHSA discoveryハング報告が
存在する([ROCm/ROCm#5812](https://github.com/ROCm/ROCm/issues/5812)、
未再現のままclose)。今回のケースは`rocminfo`自体は一瞬で成功しており、
問題はXLA/PJRTクライアント生成(ストリーム・アロケータ構築を伴う、
`rocminfo`より重い初期化)に限定される点で完全には一致しないが、
「異種デュアルGPU + RDNA4」が引き金という傾向は符合する。

### 3. 1GPUに絞ると初期化は一瞬(約1.4秒)で成功

`ROCR_VISIBLE_DEVICES=0`(RX 9070 XT / gfx1201のみ)に絞ると
`jax.devices()` → `[RocmDevice(id=0)]` が **約1.4秒** で返る。
`ROCR_VISIBLE_DEVICES=1`(RX 9060 XT側)は未検証(時間の都合で割愛、
gfx1201側が動いた時点で「異種デュアルGPU可視化が引き金」という仮説の
検証としては十分と判断)。

### 4. NEAT世代ループ: 1世代目は成功、2世代目のホスト転送でクラッシュ

`ROCR_VISIBLE_DEVICES=0` の下で、`scripts/neat_benchmark.py` のSpiralProblem
(pop=20, 2入力→1出力)を使った最小構成(`generation_limit=3`)を
`Pipeline.auto_run()` で実行:

```
compile finished, cost time: 23.093302s
Generation: 1, Cost time: 78.84ms
 	fitness: valid cnt: 20, max: -0.7124, min: -1.3879, mean: -0.9070, std: 0.1844
	node counts: max: 4, min: 3, mean: 3.10
 	conn counts: max: 3, min: 1, mean: 2.00
 	species: 1, [20]

jax.errors.JaxRuntimeError: INTERNAL: Failed to set memcpy d2d node params: HIP_ERROR_InvalidValue
```

JITコンパイル(23s)・1世代目の評価・突然変異・種分化はすべて成功し、
fitness/node数/conn数の統計も妥当な値が出ている(GPU計算自体は機能している)。
しかし2世代目に入る際の `jax.device_get()`(デバイス→ホスト転送、
内部的に `_single_device_array_to_np_array_did_copy` 経由)で
`HIP_ERROR_InvalidValue` により内部エラーがraiseされる。

**同一条件で2回実行し、同一seedで同一fitness値・同一箇所で同一エラーを確認
— 再現性のある決定論的なクラッシュ**であり、偶発的なものではない。
アプリケーション側(TensorNEAT/NEAT実装)のバグではなく、
`jax-rocm7-plugin==0.9.2` のgfx1201向けメモリコピー実装側の不具合と見られる。

## 判断への影響

`pytorch_full_port_decision_20260714.md` の非対称な賭け
(「JAX-ROCmが動けば移植不要」)は**外れた**。今回の検証で:

- GPU認識自体に異種デュアルGPU構成起因の重大なハングがある
- 仮に単一GPUに絞って回避しても、世代ループが2世代目で内部ドライバエラーで
  クラッシュする(NEATの中核である「世代を跨いだGPU計算の継続」が機能しない)

一方、PyTorch-ROCmは同一マシン上で `torch.cuda.device_count()`・
`get_device_name()`・matmul実行・順伝播ベンチマーク(Spiral 40倍、
VectorNeighbor 3.2倍)まで確認済みで、少なくとも単発の計算は安定して動く
実績がある。

**次のアクション**: `pytorch-full-port-decision` を再訪し、JAX-ROCmの
この結果(未検証→検証済み・機能しない)を踏まえて本格移植の是非を
改めて判断する。ただし移植を判断する前に、PyTorch側でも世代ループ相当の
継続的なGPU計算(複数バッチの連続実行)がクラッシュしないか、同種の
確認をしておく価値がある(現状のPyTorchプロトタイプは順伝播単発のみで、
JAXのように「2回目の呼び出しでクラッシュする」パターンを検出できていない
可能性があるため)。

## 未検証・残課題

- `ROCR_VISIBLE_DEVICES=1`(RX 9060 XT / gfx1200単体)は未検証
- `jax-rocm7-plugin` の新しいバージョン(0.10.x系)での再現有無は未確認
  (今回は既存jax/jaxlib 0.9.2とのバージョン一致を優先したため0.9.2を選定)
- `HIP_ERROR_InvalidValue` の根本原因(XLA側のGPUグラフキャプチャ機構が
  gfx1201のRDNA4アーキテクチャと非互換、等)は未調査
