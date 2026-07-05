import pytest

from query.fusion import ReciprocalRankFusion, WeightedSumFusion


def keys(ranking):
    return [key for key, _score in ranking]


def test_rrf_rewards_presence_in_both_rankings():
    fusion = ReciprocalRankFusion(k0=60)
    visual = [("a", 0.9), ("b", 0.5)]
    textual = [("b", 0.8), ("c", 0.7)]

    result = fusion.fuse([visual, textual])

    assert keys(result) == ["b", "a", "c"]
    assert result[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert result[1][1] == pytest.approx(1 / 61)
    assert result[2][1] == pytest.approx(1 / 62)


def test_rrf_ignores_score_scale():
    fusion = ReciprocalRankFusion()
    base = fusion.fuse([[("a", 0.9), ("b", 0.8)], [("b", 0.7), ("c", 0.6)]])
    scaled = fusion.fuse([[("a", 900.0), ("b", 0.001)], [("b", 7000.0), ("c", 0.0)]])

    assert base == scaled


def test_rrf_keeps_element_present_in_single_ranking():
    fusion = ReciprocalRankFusion()

    result = fusion.fuse([[("a", 0.9)], [("b", 0.8)]])

    assert keys(result) == ["a", "b"]


def test_rrf_tie_breaks_by_key():
    fusion = ReciprocalRankFusion()

    result = fusion.fuse([[("b", 1.0)], [("a", 0.5)]])

    assert keys(result) == ["a", "b"]
    assert result[0][1] == pytest.approx(result[1][1])


def test_rrf_empty_rankings():
    fusion = ReciprocalRankFusion()

    assert fusion.fuse([]) == []
    assert fusion.fuse([[], []]) == []


def test_rrf_k_truncates_result():
    fusion = ReciprocalRankFusion()
    rankings = [[("a", 0.9), ("b", 0.8), ("c", 0.7)]]

    assert keys(fusion.fuse(rankings, k=2)) == ["a", "b"]


def test_rrf_k_larger_than_candidates_returns_all():
    fusion = ReciprocalRankFusion()
    rankings = [[("a", 0.9), ("b", 0.8)]]

    assert keys(fusion.fuse(rankings, k=10)) == ["a", "b"]


def test_rrf_rejects_non_positive_k0():
    with pytest.raises(ValueError):
        ReciprocalRankFusion(k0=0)


def test_weighted_alpha_one_reproduces_first_ranking():
    fusion = WeightedSumFusion(alpha=1.0)
    visual = [("a", 0.9), ("b", 0.4), ("c", 0.1)]
    textual = [("c", 0.9), ("d", 0.2)]

    result = fusion.fuse([visual, textual])

    assert keys(result)[:3] == ["a", "b", "c"]
    assert result[3] == ("d", 0.0)


def test_weighted_alpha_zero_reproduces_second_ranking():
    fusion = WeightedSumFusion(alpha=0.0)
    visual = [("a", 0.9), ("b", 0.4)]
    textual = [("c", 0.9), ("d", 0.2)]

    result = fusion.fuse([visual, textual])

    assert keys(result)[0] == "c"
    assert keys(result).index("c") < keys(result).index("d")


def test_weighted_mid_alpha_hand_computed():
    fusion = WeightedSumFusion(alpha=0.6)
    visual = [("a", 1.0), ("b", 0.5), ("c", 0.0)]
    textual = [("c", 1.0), ("a", 0.0)]

    result = fusion.fuse([visual, textual])

    assert keys(result) == ["a", "c", "b"]
    assert result[0][1] == pytest.approx(0.6)
    assert result[1][1] == pytest.approx(0.4)
    assert result[2][1] == pytest.approx(0.3)


def test_weighted_constant_scores_normalize_to_max():
    fusion = WeightedSumFusion(alpha=0.5)

    result = fusion.fuse([[("a", 0.7), ("b", 0.7)], []])

    assert result == [("a", 0.5), ("b", 0.5)]


def test_weighted_missing_element_gets_zero_from_other_ranking():
    fusion = WeightedSumFusion(alpha=0.5)
    visual = [("a", 0.8), ("b", 0.2)]
    textual = [("c", 0.5)]

    result = fusion.fuse([visual, textual])

    assert keys(result) == ["a", "c", "b"]
    assert result[0][1] == pytest.approx(0.5)
    assert result[1][1] == pytest.approx(0.5)
    assert result[2][1] == pytest.approx(0.0)


def test_weighted_empty_rankings():
    fusion = WeightedSumFusion()

    assert fusion.fuse([[], []]) == []


def test_weighted_k_truncates_result():
    fusion = WeightedSumFusion(alpha=1.0)
    rankings = [[("a", 0.9), ("b", 0.8), ("c", 0.7)], []]

    assert keys(fusion.fuse(rankings, k=1)) == ["a"]


def test_weighted_requires_exactly_two_rankings():
    fusion = WeightedSumFusion()

    with pytest.raises(ValueError):
        fusion.fuse([[("a", 1.0)]])
    with pytest.raises(ValueError):
        fusion.fuse([[], [], []])


def test_weighted_rejects_alpha_out_of_range():
    with pytest.raises(ValueError):
        WeightedSumFusion(alpha=-0.1)
    with pytest.raises(ValueError):
        WeightedSumFusion(alpha=1.1)
