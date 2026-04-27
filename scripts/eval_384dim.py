"""
scripts/eval_384dim.py
384次元統合評価: all-MiniLM-L6-v2 実 embedding で ContextSensitiveSearch を測定

文脈あり/なしの検索結果差を 3 つの指標で定量化する:
  1. Ranking shift rate   — 文脈変化による順位変動率
  2. Domain recall@k      — 意図したドメインが top-k に入る割合
  3. Context sensitivity  — 文脈がスコアに与えた方向変化量

使い方:
  poetry run python scripts/eval_384dim.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise SystemExit("sentence-transformers が必要です: pip install sentence-transformers")

from src.med_integration.context_search import AssociationFn, ContextSensitiveSearch
from src.med_integration.hyperbolic_association import HyperbolicAssociationFn

# ──────────────────────────────────────────────
# コーパス設定
# ──────────────────────────────────────────────

SCIENCE_DOCS = [
    "犬は哺乳類に分類される脊椎動物である",
    "パブロフは犬を使って条件反射の実験を行った",
    "犬の嗅覚は人間の1万倍以上と言われている",
    "シェパードは警察犬として訓練され高い服従性を示す",
    "犬の認知能力と神経系に関する動物行動学的研究",
]

CREATIVE_DOCS = [
    "忠犬ハチ公は主人への変わらぬ忠誠心の象徴だ",
    "孤独な少年と迷い犬が友情を育む感動的な物語",
    "犬と人間の深い絆は詩や小説に繰り返し描かれてきた",
    "愛犬を失った悲しみと再生を描いた心温まる小説",
    "野良犬が人の優しさに出会い居場所を見つけるストーリー",
]

UNRELATED_DOCS = [
    "今日は晴れで気温が高く過ごしやすい",
    "微積分は数学の基礎となる重要な概念だ",
    "コーヒーの産地によって味わいと香りが異なる",
    "量子コンピュータは従来の計算機と異なる原理で動作する",
    "イタリア料理はパスタとピザが代表的なメニューだ",
]

ALL_DOCS  = SCIENCE_DOCS + CREATIVE_DOCS + UNRELATED_DOCS
SCI_IDX   = set(range(0, 5))
CRE_IDX   = set(range(5, 10))

QUERY           = "犬"
CONTEXT_SCIENCE  = "科学 研究 実験 動物学 神経科学 哺乳類 生態"
CONTEXT_CREATIVE = "物語 小説 感情 友情 忠誠心 詩 文学"


# ──────────────────────────────────────────────
# 指標計算
# ──────────────────────────────────────────────

def ranking_shift_rate(rank_a: list[int], rank_b: list[int]) -> float:
    """2 つのランキング間で順位が変わったドキュメントの割合"""
    assert len(rank_a) == len(rank_b)
    changed = sum(1 for i, (a, b) in enumerate(zip(rank_a, rank_b)) if a != b)
    return changed / len(rank_a)


def domain_recall_at_k(results: list, domain_idx: set, k: int) -> float:
    """top-k のうち指定ドメインに属するものの割合"""
    top_k = {r.index for r in results[:k]}
    hit   = top_k & domain_idx
    return len(hit) / len(domain_idx)


def rank_improvement(
    searcher: ContextSensitiveSearch,
    query_emb: np.ndarray,
    ctx_emb: np.ndarray,
    domain_idx: set,
    n_total: int,
    alpha: float = 0.5,
) -> dict:
    """
    文脈あり/なしで対象ドメインのランクがどう変わったかを返す。
    rank は 0-indexed (小さいほど上位)。
    """
    k = n_total
    res_none = searcher.search(query_emb, context_emb=None,    k=k, alpha=alpha)
    res_ctx  = searcher.search(query_emb, context_emb=ctx_emb, k=k, alpha=alpha)

    rank_none = {r.index: i for i, r in enumerate(res_none)}
    rank_ctx  = {r.index: i for i, r in enumerate(res_ctx)}

    improvements = []
    for idx in domain_idx:
        r_none = rank_none.get(idx, k)
        r_ctx  = rank_ctx.get(idx,  k)
        improvements.append(r_none - r_ctx)  # 正 = 上がった

    improved = sum(1 for d in improvements if d > 0)
    return {
        "improved": improved,
        "total":    len(domain_idx),
        "mean_delta": float(np.mean(improvements)),
    }


# ──────────────────────────────────────────────
# メイン評価
# ──────────────────────────────────────────────

def evaluate(name: str, searcher: ContextSensitiveSearch, embs: np.ndarray, q_emb: np.ndarray,
             sci_ctx: np.ndarray, cre_ctx: np.ndarray) -> None:
    K = 5
    N = len(ALL_DOCS)

    res_none = searcher.search(q_emb, context_emb=None,     k=N, alpha=0.5)
    res_sci  = searcher.search(q_emb, context_emb=sci_ctx,  k=N, alpha=0.5)
    res_cre  = searcher.search(q_emb, context_emb=cre_ctx,  k=N, alpha=0.5)

    ids_none = [r.index for r in res_none]
    ids_sci  = [r.index for r in res_sci]
    ids_cre  = [r.index for r in res_cre]

    shift_sci = ranking_shift_rate(ids_none[:10], ids_sci[:10])
    shift_cre = ranking_shift_rate(ids_none[:10], ids_cre[:10])

    sci_rec_none = domain_recall_at_k(res_none, SCI_IDX, K)
    sci_rec_sci  = domain_recall_at_k(res_sci,  SCI_IDX, K)
    cre_rec_none = domain_recall_at_k(res_none, CRE_IDX, K)
    cre_rec_cre  = domain_recall_at_k(res_cre,  CRE_IDX, K)

    ri_sci = rank_improvement(searcher, q_emb, sci_ctx, SCI_IDX, N)
    ri_cre = rank_improvement(searcher, q_emb, cre_ctx, CRE_IDX, N)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Ranking shift top-10  (ctx=science) : {shift_sci:.1%}")
    print(f"  Ranking shift top-10  (ctx=creative): {shift_cre:.1%}")
    print(f"  Science recall@{K}  no ctx → sci ctx  : "
          f"{sci_rec_none:.1%} → {sci_rec_sci:.1%}")
    print(f"  Creative recall@{K} no ctx → cre ctx  : "
          f"{cre_rec_none:.1%} → {cre_rec_cre:.1%}")
    print(f"  Rank improvement (science docs / sci ctx): "
          f"{ri_sci['improved']}/{ri_sci['total']} docs up, "
          f"mean Δrank={ri_sci['mean_delta']:+.2f}")
    print(f"  Rank improvement (creative docs / cre ctx): "
          f"{ri_cre['improved']}/{ri_cre['total']} docs up, "
          f"mean Δrank={ri_cre['mean_delta']:+.2f}")

    # ラベル付き top-5 表示 (S=science, C=creative, -=unrelated)
    def label(idx: int) -> str:
        if idx in SCI_IDX: return "S"
        if idx in CRE_IDX: return "C"
        return "-"

    def fmt_top5(res: list) -> str:
        return "  ".join(f"[{label(r.index)}]{ALL_DOCS[r.index][:12]}" for r in res[:5])

    print(f"\n  Top-5 (no ctx) : {fmt_top5(res_none)}")
    print(f"  Top-5 (sci ctx): {fmt_top5(res_sci)}")
    print(f"  Top-5 (cre ctx): {fmt_top5(res_cre)}")


# ──────────────────────────────────────────────
# シナリオ2: 英語コーパス (曖昧クエリ "python")
# ──────────────────────────────────────────────

PROG_DOCS = [
    "Python is a high-level programming language with clear syntax",
    "Python supports object-oriented and functional programming paradigms",
    "Python is widely used in machine learning and data science",
    "Flask and Django are popular Python web frameworks",
    "Python list comprehensions provide concise data manipulation",
]
ANIMAL_DOCS = [
    "Python regius, the ball python, is a popular pet snake",
    "Burmese pythons are large constrictors native to Southeast Asia",
    "Pythons are ambush predators that kill prey by constriction",
    "Python molurus can reach lengths of over 5 meters",
    "Ball pythons curl into a ball as a defensive behavior",
]
UNRELA_DOCS_EN = [
    "The stock market closed higher on Friday",
    "Scientists discover a new exoplanet in the habitable zone",
    "A new recipe for homemade pasta with fresh tomatoes",
    "The history of ancient Rome and its empire",
    "Climate change impacts on Arctic ice sheets",
]

ALL_EN   = PROG_DOCS + ANIMAL_DOCS + UNRELA_DOCS_EN
PROG_IDX = set(range(0, 5))
ANIM_IDX = set(range(5, 10))

QUERY_EN    = "python"
CTX_PROG    = "software programming code algorithm developer"
CTX_ANIMAL  = "snake reptile wildlife biology constrictor"


def evaluate_english(name: str, model, fn_cls) -> None:
    doc_embs = model.encode(ALL_EN, normalize_embeddings=True)
    q_emb    = model.encode([QUERY_EN], normalize_embeddings=True)[0]
    ctx_prog = model.encode([CTX_PROG],   normalize_embeddings=True)[0]
    ctx_anim = model.encode([CTX_ANIMAL], normalize_embeddings=True)[0]

    from src.med_integration.context_search import ContextSensitiveSearch
    searcher = ContextSensitiveSearch(association_fn=fn_cls())
    searcher.build_index(doc_embs.astype(np.float32), ALL_EN)

    N = len(ALL_EN)
    res_none = searcher.search(q_emb, context_emb=None,     k=N, alpha=0.5)
    res_prog = searcher.search(q_emb, context_emb=ctx_prog, k=N, alpha=0.5)
    res_anim = searcher.search(q_emb, context_emb=ctx_anim, k=N, alpha=0.5)

    ri_prog = rank_improvement(searcher, q_emb, ctx_prog, PROG_IDX, N)
    ri_anim = rank_improvement(searcher, q_emb, ctx_anim, ANIM_IDX, N)

    def label(idx):
        if idx in PROG_IDX: return "P"
        if idx in ANIM_IDX: return "A"
        return "-"

    def fmt5(res):
        return "  ".join(f"[{label(r.index)}]{ALL_EN[r.index][:18]}" for r in res[:5])

    K = 5
    print(f"\n{'='*60}")
    print(f"  [EN] {name}  query='python'")
    print(f"{'='*60}")
    print(f"  Prog recall@{K}   no ctx → prog ctx : "
          f"{domain_recall_at_k(res_none, PROG_IDX, K):.1%} → "
          f"{domain_recall_at_k(res_prog, PROG_IDX, K):.1%}")
    print(f"  Animal recall@{K} no ctx → anim ctx : "
          f"{domain_recall_at_k(res_none, ANIM_IDX, K):.1%} → "
          f"{domain_recall_at_k(res_anim, ANIM_IDX, K):.1%}")
    print(f"  Rank improvement (prog docs / prog ctx): "
          f"{ri_prog['improved']}/{ri_prog['total']} docs up, "
          f"mean Δrank={ri_prog['mean_delta']:+.2f}")
    print(f"  Rank improvement (animal docs / anim ctx): "
          f"{ri_anim['improved']}/{ri_anim['total']} docs up, "
          f"mean Δrank={ri_anim['mean_delta']:+.2f}")
    print(f"\n  Top-5 (no ctx)   : {fmt5(res_none)}")
    print(f"  Top-5 (prog ctx) : {fmt5(res_prog)}")
    print(f"  Top-5 (anim ctx) : {fmt5(res_anim)}")


def main() -> None:
    print("モデルをロード中: all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("embedding を計算中 ...")
    doc_embs = model.encode(ALL_DOCS, normalize_embeddings=True)
    q_emb    = model.encode([QUERY], normalize_embeddings=True)[0]
    sci_ctx  = model.encode([CONTEXT_SCIENCE],  normalize_embeddings=True)[0]
    cre_ctx  = model.encode([CONTEXT_CREATIVE], normalize_embeddings=True)[0]

    print(f"embedding 次元: {doc_embs.shape[1]}")
    print(f"コーパス: {len(ALL_DOCS)} 件 "
          f"(science={len(SCI_IDX)}, creative={len(CRE_IDX)}, unrelated={len(UNRELATED_DOCS)})")

    # ── MLP AssociationFn ──────────────────────
    mlp_fn  = AssociationFn()
    mlp_s   = ContextSensitiveSearch(association_fn=mlp_fn)
    mlp_s.build_index(doc_embs.astype(np.float32), ALL_DOCS)
    evaluate("MLP AssociationFn (w=[0.25,0.25,0.25,0.25])", mlp_s, doc_embs,
             q_emb, sci_ctx, cre_ctx)

    # ── Hyperbolic AssociationFn ───────────────
    hyp_fn  = HyperbolicAssociationFn()
    hyp_s   = ContextSensitiveSearch(association_fn=hyp_fn)
    hyp_s.build_index(doc_embs.astype(np.float32), ALL_DOCS)
    evaluate("HyperbolicAssociationFn (c=1.0, ctx_weight=0.3)", hyp_s, doc_embs,
             q_emb, sci_ctx, cre_ctx)

    # ── 英語シナリオ (曖昧クエリ "python") ─────
    print("\n\n--- 英語シナリオ: 曖昧クエリ 'python' ---")
    print("(programming vs animal で文脈感度を測定)")
    evaluate_english("MLP AssociationFn",        model, AssociationFn)
    evaluate_english("HyperbolicAssociationFn",  model, HyperbolicAssociationFn)

    print("\n" + "="*60)
    print("  評価完了")
    print("="*60)


if __name__ == "__main__":
    main()
