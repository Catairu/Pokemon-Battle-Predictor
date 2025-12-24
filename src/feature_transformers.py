import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from math import prod

# ==============================================================================
# 1. HELPER FUNCTIONS
# ==============================================================================


def hp_features(battle_timeline):
    """
    Calculates the total HP difference (Team 1 - Team 2) at the end of the battle.

    This is based on the last recorded HP percentage for each Pokemon in the
    battle_timeline.
    """
    team1 = {}
    team2 = {}
    for turn in battle_timeline:
        team1[turn["p1_pokemon_state"]["name"]] = turn["p1_pokemon_state"]["hp_pct"]
        team2[turn["p2_pokemon_state"]["name"]] = turn["p2_pokemon_state"]["hp_pct"]

    team1_hp = 6 - sum((1 - hp) for hp in team1.values())
    team2_hp = 6 - sum((1 - hp) for hp in team2.values())

    return team1_hp, team2_hp


def status_features(battle_timeline, num_turns=30):
    """
    Calculates the status proportions for P1 and P2, along with their differences.
    Tracks the proportion of turns in which P1 and P2 were affected by 'par', 'frz', or 'slp'
    """

    all_status = ["par", "frz", "slp"]

    status_1 = {s: 0 for s in all_status}
    status_2 = {s: 0 for s in all_status}

    for turn in battle_timeline:
        s1 = turn["p1_pokemon_state"].get("status")
        s2 = turn["p2_pokemon_state"].get("status")
        if s1 in all_status:
            status_1[s1] += 1
        if s2 in all_status:
            status_2[s2] += 1

    prop_1 = {s: status_1[s] / num_turns for s in all_status}
    prop_2 = {s: status_2[s] / num_turns for s in all_status}

    features = {}
    for s in all_status:
        features[f"p1_{s}"] = prop_1[s]
        features[f"p2_{s}"] = prop_2[s]
        features[f"diff_{s}"] = prop_2[s] - prop_1[s]

    return features


def none_features(battle_timeline):
    """
    Calculates the difference in the percentage of turns without moves between p2 and p1.
    A value > 0 means that p2 had a higher percentage of “None” turns compared to p1.
    """
    p1_none_count = 0
    p2_none_count = 0

    for turn in battle_timeline:
        if not turn.get("p1_move_details"):
            p1_none_count += 1
        if not turn.get("p2_move_details"):
            p2_none_count += 1

    p1_none_rate = p1_none_count / 30
    p2_none_rate = p2_none_count / 30

    return p1_none_rate, p2_none_rate, p2_none_rate - p1_none_rate


def compute_move_features(battle_timeline):
    """
    Calculates mean move power and accuracy for p1 and p2,
    plus the differences between them (p1 - p2).

    'None' moves (no move details) are treated as having 0 power
    and 0 accuracy in the mean calculation.
    """

    stats = {
        "p1": {"base_power": [], "accuracy": []},
        "p2": {"base_power": [], "accuracy": []},
    }

    for turn in battle_timeline:
        for pid in ["p1", "p2"]:
            move = turn.get(f"{pid}_move_details")
            if move:
                stats[pid]["base_power"].append(move.get("base_power", 0) or 0)
                stats[pid]["accuracy"].append(move.get("accuracy", 0) or 0)
            else:
                stats[pid]["base_power"].append(0)
                stats[pid]["accuracy"].append(0)

    features = {}
    for pid in ["p1", "p2"]:
        bp = np.array(stats[pid]["base_power"], dtype=float)
        acc = np.array(stats[pid]["accuracy"], dtype=float)

        features[f"{pid}_mean_power"] = bp.mean()
        features[f"{pid}_mean_acc"] = acc.mean()

    features["diff_mean_power"] = (
        features["p1_mean_power"] - features["p2_mean_power"]
    ) / 100.0
    features["diff_mean_acc"] = features["p1_mean_acc"] - features["p2_mean_acc"]
    return features


def extract_battle_features(timeline):
    """
    Extracts battle features (HP lost/gained, Pokémon seen/fainted)
    for P1 and P2 from a battle timeline.
    Returns a dictionary, ideal for DataFrame.apply().
    """

    p1_total_hp_lost = 0.0
    p1_total_hp_gained = 0.0
    p1_last_hp_pcts = {}

    p2_total_hp_lost = 0.0
    p2_total_hp_gained = 0.0
    p2_last_hp_pcts = {}

    p1_seen_names = set()
    p2_seen_names = set()

    p1_fainted_names = set()
    p2_fainted_names = set()

    for turn in timeline:

        p1_state = turn.get("p1_pokemon_state")
        if p1_state:
            name = p1_state["name"]
            current_hp = p1_state["hp_pct"]

            p1_seen_names.add(name)

            if name in p1_last_hp_pcts:
                previous_hp = p1_last_hp_pcts[name]
                hp_change = current_hp - previous_hp

                if hp_change < 0:
                    p1_total_hp_lost += abs(hp_change)
                elif hp_change > 0:
                    p1_total_hp_gained += hp_change
            else:
                p1_total_hp_lost += 1.0 - current_hp

            p1_last_hp_pcts[name] = current_hp

            if current_hp == 0.0:
                p1_fainted_names.add(name)

        p2_state = turn.get("p2_pokemon_state")
        if p2_state:
            name = p2_state["name"]
            current_hp = p2_state["hp_pct"]
            p2_seen_names.add(name)

            if name in p2_last_hp_pcts:
                previous_hp = p2_last_hp_pcts[name]
                hp_change = current_hp - previous_hp

                if hp_change < 0:
                    p2_total_hp_lost += abs(hp_change)
                elif hp_change > 0:
                    p2_total_hp_gained += hp_change
            else:
                p2_total_hp_lost += 1.0 - current_hp

            p2_last_hp_pcts[name] = current_hp

            if current_hp == 0.0:
                p2_fainted_names.add(name)

    features = {
        "p1_hp_lost": round(p1_total_hp_lost, 4),
        "p1_hp_gained": round(p1_total_hp_gained, 4),
        "p1_pokemon_seen": len(p1_seen_names),
        "p1_pokemon_fainted": len(p1_fainted_names),
        "p2_hp_lost": round(p2_total_hp_lost, 4),
        "p2_hp_gained": round(p2_total_hp_gained, 4),
        "p2_pokemon_seen": len(p2_seen_names),
        "p2_pokemon_fainted": len(p2_fainted_names),
    }

    return features


# ==============================================================================
# 2. CUSTOM SKLEARN TRANSFORMERS
# ==============================================================================


class HPFeaturesExtractor(BaseEstimator, TransformerMixin):
    """Extracts HP-related features from a battle timeline."""

    def __init__(self, timeline_column="battle_timeline"):
        self.timeline_column = timeline_column
        self.feature_names = ["p1_final_hp", "p2_final_hp"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        hp_results = X_transformed[self.timeline_column].apply(hp_features)
        hp_df = pd.DataFrame(
            hp_results.tolist(), columns=self.feature_names, index=X_transformed.index
        )
        X_transformed = pd.concat([X_transformed, hp_df], axis=1)
        X_transformed["hp_difference"] = (
            X_transformed["p1_final_hp"] - X_transformed["p2_final_hp"]
        )
        X_transformed["hp_ratio"] = X_transformed["p1_final_hp"] / (
            X_transformed["p2_final_hp"] + 1e-5
        )
        return X_transformed


class StatusDiffExtractor(BaseEstimator, TransformerMixin):
    """Adds normalized status condition proportions and differences as features."""

    def __init__(self, timeline_column="battle_timeline"):
        self.timeline_column = timeline_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        status_results = (
            X_transformed[self.timeline_column].apply(status_features).apply(pd.Series)
        )
        return pd.concat([X_transformed, status_results], axis=1)


class MissDiffExtractor(BaseEstimator, TransformerMixin):
    """
    Calls the external function none_features to obtain the rates of p1 and p2, and computes their difference.
    """

    def __init__(self, timeline_column="battle_timeline"):
        self.timeline_column = timeline_column
        self.feature_names = ["p1_none_rate", "p2_none_rate", "none_move_diff"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        results_series = X_transformed[self.timeline_column].apply(none_features)
        miss_df = pd.DataFrame(
            results_series.tolist(),
            columns=self.feature_names,
            index=X_transformed.index,
        )
        return pd.concat([X_transformed, miss_df], axis=1)


class MoveFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer that extracts move-based statistical features
    from a column containing battle timelines.
    """

    def __init__(self, timeline_column: str = "battle_timeline"):
        self.timeline_column = timeline_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        move_features = (
            X_transformed[self.timeline_column]
            .apply(compute_move_features)
            .apply(pd.Series)
        )
        return pd.concat([X_transformed, move_features], axis=1)


class SwitchFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom feature extractor that applies the switch_features() function to each battle,
    adds the resulting features to the dataset, and returns the transformed DataFrame.
    """

    def __init__(self, timeline_column="battle_timeline"):
        self.timeline_column = timeline_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()

        switch_data = [
            switch_features(timeline) for timeline in X_transformed["battle_timeline"]
        ]
        X_transformed["p1_switches"] = [s[0] for s in switch_data]
        X_transformed["p2_switches"] = [s[1] for s in switch_data]
        X_transformed["diff_switches"] = (
            X_transformed["p1_switches"] - X_transformed["p2_switches"]
        )

        return X_transformed


class BattleFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer that extracts battle statistics (HP, seen, fainted)
    """

    def __init__(self, timeline_column: str = "battle_timeline"):
        self.timeline_column = timeline_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        battle_features = (
            X_transformed[self.timeline_column]
            .apply(extract_battle_features)
            .apply(pd.Series)
        )
        return pd.concat([X_transformed, battle_features], axis=1)


class DropColumnTransformer(BaseEstimator, TransformerMixin):
    """Drops specified columns from the DataFrame."""

    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols_to_drop = [col for col in self.columns_to_drop if col in X.columns]
        return X.drop(columns=cols_to_drop)
