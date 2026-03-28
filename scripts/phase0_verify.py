"""
Phase 0 動作確認スクリプト
TRIDENT — TensorNEAT + QDax 環境チェック
"""

import sys

results = {}

# 1. JAX
print("=" * 50)
print("1. JAX 確認")
try:
    import jax
    import jax.numpy as jnp
    devices = jax.devices()
    print(f"  version : {jax.__version__}")
    print(f"  devices : {devices}")
    gpu_available = any("gpu" in str(d).lower() or "cuda" in str(d).lower() for d in devices)
    print(f"  GPU     : {'✅ 利用可能' if gpu_available else '❌ CPU のみ（CUDA設定要確認）'}")
    results["jax"] = True
except Exception as e:
    print(f"  ❌ エラー: {e}")
    results["jax"] = False

# 2. TensorNEAT
print("\n2. TensorNEAT 確認")
try:
    import tensorneat
    print(f"  インポート : ✅")

    from tensorneat import Pipeline
    print(f"  Pipeline  : ✅")

    from tensorneat.algorithm.hyperneat import HyperNEAT
    print(f"  HyperNEAT : ✅")

    from tensorneat.genome import DefaultGenome, RecurrentGenome
    print(f"  Genomes   : ✅ (DefaultGenome, RecurrentGenome)")

    results["tensorneat"] = True
except Exception as e:
    print(f"  ❌ エラー: {e}")
    results["tensorneat"] = False

# 3. QDax MAP-Elites
print("\n3. QDax (MAP-Elites) 確認")
try:
    from qdax.core.map_elites import MAPElites
    print(f"  MAPElites : ✅")

    from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
    print(f"  Repertoire: ✅")

    results["qdax"] = True
except Exception as e:
    print(f"  ❌ エラー: {e}")
    results["qdax"] = False

# 4. evosax
print("\n4. evosax 確認")
try:
    import evosax
    print(f"  インポート : ✅ (Evolution Strategies 参考用)")
    print(f"  NS 実装   : ❌ なし → カスタム実装が必要")
    results["evosax"] = True
except Exception as e:
    print(f"  ❌ エラー: {e}")
    results["evosax"] = False

# 5. FAISS (既存 MED コンポーネント)
print("\n5. FAISS 確認（MED 既存コンポーネント）")
try:
    import faiss
    print(f"  version : {faiss.__version__}")
    print(f"  FAISS   : ✅")
    results["faiss"] = True
except ImportError:
    print(f"  FAISS   : ❌ 未インストール（MED 統合時に必要）")
    results["faiss"] = False

# サマリー
print("\n" + "=" * 50)
print("Phase 0 確認サマリー")
print("=" * 50)
all_core = results.get("jax") and results.get("tensorneat") and results.get("qdax")
for k, v in results.items():
    mark = "✅" if v else "❌"
    print(f"  {mark} {k}")

print()
if all_core:
    print("✅ コアスタック（JAX + TensorNEAT + QDax）準備完了")
    print("   Phase 1 (A型 NeatIndexer) を開始できます")
else:
    print("❌ 未解決の依存あり — 上記エラーを確認してください")

sys.exit(0 if all_core else 1)
