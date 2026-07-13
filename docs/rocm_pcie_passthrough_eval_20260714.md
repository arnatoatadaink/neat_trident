# 検討記録: ROCm-on-WSL2ブロッカーへの対応方法（PCIeパススルーVM案は保留）

日付: 2026-07-14
関連タスク: `pytorch-rocm-bench-verify`, `pytorch-full-port-decision`
関連セッション: 40db08fe-ddb7-4825-ad41-74cb1f095111（2026-07-13）

## 背景

前回セッション（40db08fe, 2026-07-13）で、WSL2上でのROCm版PyTorch実行が
ブロックされていることが判明した。詳細は以下の通り。

### 判明した事実

1. **ROCmコア（HSA/HIP）自体は正常動作** — `rocminfo`は`gfx1201`(RX 9070 XT)
   /`gfx1200`(RX 9060 XT)を正しく列挙できる。ROCm 7.2.3のシステムHSAランタイムは
   WSL用の特別な互換パス（`"WSL environment detected"`）を持っており、これが
   GPU認識できている理由。
2. **PyTorch同梱のrocprofiler-sdkがクラッシュする** — `torch.cuda.device_count()`
   呼び出し時、内部で使われるrocprofiler-sdkが`/sys/class/kfd/kfd/topology/nodes`
   （WSL2には存在しない。WSL2は`/dev/dxg`経由でGPUをパラバーチャライズしており、
   ネイティブのamdgpuカーネルドライバがロードされないため）の存在を前提に
   エージェント列挙を行い、「3 HSAエージェント検出 vs KFDトポロジー0件」の
   不整合でハードアボート（FATAL、core dump）する。
   システム側ROCm(`rocminfo`)にあるWSL互換コードが、torch同梱コンポーネントには
   存在しないというギャップ。
3. **環境変数での回避は不可** — `HSA_TOOLS_LIB=0`、`ROCPROFILER_REGISTER_ENABLE=0`
   等を試したが、いずれも`agent.cpp`内のFATALログ経路をゲートせず効果なし。
   torch同梱`librocprofiler-sdk.so`の一時無効化も試したが、システム側の同名
   ライブラリを再度読みに行く挙動で同じ場所でクラッシュ、解消せず。

### 前回時点の提案

根本原因がWSL2の仮想化GPUモデル（`/dev/dxg`、KFD sysfsなし）である以上、
ネイティブLinux（デュアルブートや別マシン）での検証が最もクリーンな回避策、
という結論だった。

## 今回の検討: PCIeパススルーLinux VM案

### 評価

ゲストLinux VMにGPUをPCIeパススルー（VFIO/DDA）すれば、ゲスト側で本物の
amdgpuカーネルドライバがロードされ`/sys/class/kfd/kfd/topology/nodes`が
正しくpopulateされるため、上記の不整合は原理的に発生しない。
「別マシンでのネイティブLinux」と技術的には等価な解決になり、かつ
RX 9070 XT / RX 9060 XTの2枚構成なら1枚をVM専有・1枚をWindowsホスト用に
残すことでデュアルブートのようなOS切り替えの手間も避けられる、という利点がある。

選択肢として Hyper-V + DDA、またはデュアルブート＋KVM/QEMU + VFIO を検討した。

### 判断: 保留

**GPUをBIOSレベルで固定割り当て（IOMMU分離・vfio-pciバインド）するコストが
見合わないため、この案は保留とする。** ハードウェア構成変更・検証の手間に対して、
得られる効果（ROCmベンチ数値の取得）が釣り合わない。

## 次の一手: rocprofiler-sdk無効化インストールの検証

パススルーVMより低コストな代替パスとして、**rocprofiler-sdkを無効化した状態で
torchをインストールできないか**を検証する。方向性の候補:

- ROCm版torchホイールのビルド/インストール時にrocprofiler-sdk関連コンポーネントを
  除外できるオプションがあるか調査（pip installのextras、環境変数、ビルドフラグ等）
- あるいはインストール後にrocprofiler-sdk関連の共有ライブラリ自体を除去/置換し、
  torch側がそれをロードしようとしないよう仕向けられるか（前回試した「一時無効化」は
  システム側の同名ライブラリを拾ってしまったため、除去の徹底度が足りなかった可能性がある）

これがWSL2上でのROCm PyTorch実行の唯一の残された低コスト経路。

## TODO登録

- `rocm-pcie-passthrough-vm-eval`（P4, pending）— PCIeパススルーVM案。
  BIOS固定割り当てコストが高いため保留。ハードウェア構成が変わる場合に再検討。
- `pytorch-rocprofiler-sdk-disable-install`（P3, pending）— rocprofiler-sdkを
  無効化した状態でのROCm版torchインストールを試す。上記の代替パス。

## 検証結果 (2026-07-14 追記): 環境変数一つで解決

`pytorch-rocprofiler-sdk-disable-install` を実施。結論は **ライブラリの除去/置換は不要、
`ROCPROFILER_REGISTER_ENABLED=0` を実行時の環境変数として渡すだけでクラッシュを回避できる**。

### 原因の再整理

クラッシュは torch 同梱の `librocprofiler-sdk.so` 自身の `agent.cpp` 内、HSAエージェント数(3)と
KFDトポロジーsysfs由来のノード数(0, WSL2には存在しないため)の不整合を検知した `FATAL` ログ
(`agent.cpp:1093`)によるアボートで、`rocprofiler_register_library_api_table` 経由で
`librocprofiler-register.so` がSDKの初期化を呼び出す経路で発生していた。

### 効かなかったもの（前回・今回とも再確認）

- `HSA_TOOLS_LIB=0`
- `ROCPROFILER_REGISTER_ENABLE=0`（**綴りミス**: 正しくは末尾に `D` が付く
  `ROCPROFILER_REGISTER_ENABLED`。前回の失敗はこの誤字が原因だった可能性が高い）
- `ROCPROFILER_REGISTER_FORCE_LOAD=0`
- `ROCPROFILER_REGISTER=0`
- `HSA_TOOLS_REPORT_LOAD_FAILURE=0`
- 同梱 `librocprofiler-sdk.so` の一時無効化（システム側 `/opt/rocm-7.2.3/lib/librocprofiler-sdk.so`
  を再度拾ってしまい同じ場所でクラッシュ、解消せず — 変わらず）

### 効いたもの

`librocprofiler-register.so` の文字列解析 (`strings`) から発見した
`ROCPROFILER_REGISTER_ENABLED` を `0` に設定すると、レジストリ自体が無効化され
SDKの `agent.cpp` 初期化ルートに到達しなくなる。

```bash
ROCPROFILER_REGISTER_ENABLED=0 poetry run python -c "
import torch
print(torch.cuda.device_count())      # -> 2
print(torch.cuda.is_available())      # -> True
print(torch.cuda.get_device_name(0))  # -> AMD Radeon RX 9070 XT
print(torch.cuda.get_device_name(1))  # -> AMD Radeon RX 9060 XT
"
```

`cuda`デバイス上でのmatmul (`torch.randn(1000,1000, device='cuda') @ ...`) も実行・同期成功を確認済み
（単なるdevice_count()だけでなく実計算パスが通ることを確認）。

**sudo不要・ライブラリの除去/置換不要**。次の一手 (`pytorch-rocm-bench-verify`) では
ベンチマーク実行時に `ROCPROFILER_REGISTER_ENABLED=0` を環境変数として付与すればよい。

## pytorch-rocm-bench-verify 実施結果 (2026-07-14)

`ROCPROFILER_REGISTER_ENABLED=0 poetry run python scripts/pytorch_neat_prototype_benchmark.py`
をRX 9070 XT (torch 2.13.0+rocm7.2) 上で実行。結果は
`logs/pytorch_proto_benchmark.jsonl` に追記済み。CPU実行結果（同ログ内の直近値）との比較:

| label | CPU mean_forward_ms | ROCm(cuda) mean_forward_ms | 速度向上 |
|---|---|---|---|
| Spiral (pop=100, samples=200) | 1268.083 | 31.303 | 約40.5倍 |
| VectorNeighbor (pop=100, samples=20) | 67.65 | 21.069 | 約3.2倍 |

`poetry run pytest tests/test_pytorch_proto.py -v` は3件とも pass（既存実装は非破壊）。

**結論**: ROCm実機でのGPUオフロードは明確に有効（順伝播のみの計測、下界比較）。
Spiralは低次元入力(2→1)にもかかわらず40倍超の高速化が出ており、GPUバッチ化の
恩恵がノード/コネクション数の小さいネットワークでも十分に出ることを確認。
VectorNeighborは相対的に伸びが小さい（3.2倍）— 高次元入出力(32→32)でのCPU側の
ベクトル化効率が既に高いためと推測されるが、詳細分析は未実施。

次のタスク `pytorch-full-port-decision`（PyTorch本格移植の可否判断）の判断材料として使う。
