# 決定記録: フルテストスイート未完走問題への対応 (jax-rocm-fulltest-workaround-revisit)

日付: 2026-07-14
関連タスク: `jax-rocm-fulltest-workaround-revisit`
関連文書: `docs/pytorch_map_elites_replacement_spike_20260714.md`(2章、本問題の初報告),
`docs/jax_rocm_spike_20260714.md`(GPU側の根本原因の実機検証)
成果物: `tests/conftest.py`, `.claude/rules/jax-verification.md`(更新)

## 結論: `tests/conftest.py` で `JAX_PLATFORMS=cpu` をデフォルト化 — 元の3案とは別の第4の案を採用

`poetry run pytest tests/ -q` を実行するテストコード自体はNEAT/ES-HyperNEATの
アルゴリズム正しさを検証するものであり、GPUハードウェアの実機動作検証ではない。
そのため「GPU上で動かして問題を回避する」方向の3案（ROCR_VISIBLE_DEVICES明記/
gpuマーカー分離/プロセス分離）を検討する前提そのものを疑い、CPUバックエンドで
テストを実行することで問題を根本から回避できないか確認した。

**確認できた**: `JAX_PLATFORMS=cpu` を設定すると、ROCmプラグインの初期化自体が
スキップされ、デバイス列挙ハングも世代ループ2世代目のクラッシュも発生しない。
`tests/conftest.py` に `os.environ.setdefault("JAX_PLATFORMS", "cpu")` を追加し
(setdefaultなので `JAX_PLATFORMS=rocm poetry run pytest ...` で上書き可能)、
`poetry run pytest tests/ -q` を実行 → **204 passed, 0 failed, 0 errors, 193.47s**
(JAX/CUDA JITコンパイルが発生しないため、GPU実行時より高速)。

## 上流issue調査

以下について GitHub issue tracker (ROCm/ROCm, ROCm/rocm-jax, jax-ml/jax,
ROCm/rocm-systems, vllm-project/vllm) を調査した。

### 異種デュアルGPU (gfx1200+gfx1201) でのデバイス列挙ハング
- [ROCm/ROCm#5812](https://github.com/ROCm/ROCm/issues/5812) —
  RDNA4デュアルGPU構成でのHSA discoveryハング報告。再現できず一度closeされている。
  `jax_rocm_spike_20260714.md` で確認した症状（`rocminfo`自体は成功するが
  PJRTクライアント生成でハング）とは完全には一致しないが、
  「異種デュアルGPU + RDNA4」が引き金という傾向は符合する。
- [ROCm/rocm-jax#390](https://github.com/ROCm/rocm-jax/issues/390) —
  マルチデバイス環境でGPU 0のname/architectureが全GPUに対して報告される
  既知バグ。JAX-ROCmのマルチGPU識別ロジック自体に既知の不具合があることを示す
  傍証(直接の原因一致ではない)。
- [ROCm/rocm-systems#5480](https://github.com/ROCm/rocm-systems/issues/5480) —
  RCCL 2.27.7でのgfx1201デュアルGPU間P2P通信デッドロック(vLLM TP=2)。
  **未解決(2026-04-27オープン、担当者アサイン済みだが未修正)**。
  P2P不可時のホストステージングへのフォールバックが壊れている可能性が
  指摘されている。ただし本件はマルチGPU P2P通信に固有の問題であり、
  今回の`HIP_ERROR_InvalidValue`クラッシュは`ROCR_VISIBLE_DEVICES=0`で
  単一GPUに絞った状態でも再現するため、直接の同一原因ではない
  （デュアルGPU環境全般でRDNA4のROCmサポートが未成熟、という傾向の裏付けとして扱う）。

### 単一GPU・世代ループ2世代目の `HIP_ERROR_InvalidValue`(memcpy d2d node params)
- 完全一致するissueは見つからなかった。`jax-rocm7-plugin==0.9.2` /
  `ROCm/rocm-jax` のissueトラッカーを検索したが、該当のエラー文字列
  ("Failed to set memcpy d2d node params"、HIPグラフのmemcpyノード関連)を
  報告した既知issueはない。HIPグラフAPI一般のmemcpyノード関連バグ
  ([ROCm/hip#1245](https://github.com/ROCm/hip/issues/1245) 等)は存在するが
  gfx1201・JAX-ROCmとの関連は確認できなかった。
- **結論**: 未報告のニッチな不具合である可能性が高く、上流での近い将来の
  修正を前提にできない。3案の再考において「待てば直る」という選択肢は
  棄却材料として扱う。

## 元の3案の再評価

1. **`ROCR_VISIBLE_DEVICES=0 ROCPROFILER_REGISTER_ENABLED=0` を前提条件として明記**:
   ハング自体は解消するが、世代ループ2世代目のクラッシュ(52 errors/6 failed)は
   解消しない。単独では `poetry run pytest tests/ -q` を完走させる案として不十分。
2. **JAX-GPU実機テストに`gpu`マーカーを切り別運用にする**:
   デフォルト実行からGPU依存テストを除外すればスイートは緑になるが、
   実際に検証してみると204件すべてがCPUバックエンドで**そのまま合格する**
   ことが判明した（後述）。つまり「除外」ではなくテスト対象を変えずに
   バックエンドだけCPUに倒せば、除外なしで全カバレッジを維持できる。
   マーカー分離は「本当にGPU実機でしか検証できないテスト」が今後追加された
   場合の受け皿として概念上は残すが、現時点でそれに該当するテストは
   スイート内に存在しない。
3. **`pytest-forked`等でtorch系/jax系プロセスを分離**:
   `pytorch_map_elites_replacement_spike_20260714.md` の切り分け実験で
   「torch系ファイルを`--ignore`してもハングは再現する」ことが既に確認されて
   おり、原因はtorchとの同一プロセス共存ではなくJAX-ROCm単体の問題
   （デバイス列挙・世代ループ内メモリ転送）。クラッシュはtest間ではなく
   *1つのtestプロセス内の世代ループ内*で発生するため、test単位のプロセス
   分離(`pytest-forked`)ではそもそも防げない。依存追加のコストに見合う効果が
   ないため棄却。

## 採用した案(元の3案の外)

**テストスイートのデフォルトバックエンドをCPUにする。**
`tests/conftest.py` で `os.environ.setdefault("JAX_PLATFORMS", "cpu")` を設定
— テストモジュールが `import jax` するより前に効かせる必要があるため
conftest.pyのトップレベルに置いた。`setdefault`により
`JAX_PLATFORMS=rocm poetry run pytest tests/ -q` で意図的にGPU実行へ
上書き可能。

根拠:
- テストスイートの目的はNEAT/ES-HyperNEATのアルゴリズム正しさの検証であり、
  GPU実機の演算結果を検証するものではない(GPU実機のベンチマーク・検証は
  `scripts/neat_benchmark.py` 等が別途担っている — `CLAUDE.md` のCommands表参照)。
- 案2(gpuマーカー分離)は「除外前提」だったが、実際には除外不要で全204件が
  CPUで合格した。案1・3が対処しようとしていた「GPUで動かす」という前提
  そのものが不要だった。
- 上流での修正時期が見通せない(該当issue未報告)ため、GPU実機での完走を
  前提にした運用は当面成立しない。

## 検証結果

```
$ poetry run pytest tests/ -q
........................................................................ [ 35%]
........................................................................ [ 70%]
............................................................             [100%]
204 passed, 4 warnings in 193.47s (0:03:13)
```

環境変数を明示的に渡さず(`tests/conftest.py`のみで)完走を確認。
ハングなし・クラッシュなし・全件合格。

## 副作用・残課題

- テストスイートはCPU実行のみで担保されるようになり、GPU実機での
  「NEAT世代ループを継続的にGPU上で回せるか」という検証はテストスイートの
  スコープ外になる(元々JAX側はこの検証に失敗する状態だったため、実質的な
  後退ではない — `jax_rocm_spike_20260714.md`参照)。将来GPU実機の継続動作を
  スイートに組み込みたくなった場合は、`gpu`マーカー + 上記env var前提条件を
  追加する形で拡張可能。
- 上流issue(`ROCm/rocm-jax`, `ROCm/rocm-systems#5480`等)は監視対象として
  残すが、本タスクでの追跡専用issueの新規起票は行っていない。
