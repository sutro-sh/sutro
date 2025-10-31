from typing import Union, List, Tuple
import polars as pl
import pandas as pd
import numpy as np
import math
from ..common import ModelOptions
from ..interfaces import BaseSutroClient
from collections import defaultdict
from itertools import combinations


class Score(BaseSutroClient):
    def score(
        self,
        data: Union[List, pd.DataFrame, pl.DataFrame, str],
        model: ModelOptions = "gemma-3-12b-it",
        job_priority: int = 0,
        name: Union[str, List[str]] = None,
        description: Union[str, List[str]] = None,
        column: Union[str, List[str]] = None,
        # function-specific parameters
        criteria: Union[str, List[str]] = None,
        score_column_name: str = "score",
        range: Tuple[int, int] = (0, 10),
    ):
        """
        A simple invocation of an LLM-as-a-judge numerical scoring function, with a default 0-10 range.

        Accepts a normal Sutro input data type, as well as a string or list of strings to use as criteria,
        a column name to use for the scoring, and a range to use for the scoring (default 0-10).

        Returns a pandas or polars dataframe with the scores as a column, corresponding to the column name provided.
        """

        if isinstance(criteria, str):
            criteria = [criteria]

        system_prompt = f"""You are a judge. Your job is to score the data presented to you according to the following criteria:
{", ".join(criteria)}
Return a score between {range[0]} and {range[1]}, and nothing else."""

        json_schema = {
            "type": "object",
            "properties": {
                f"{score_column_name}": {
                    "type": "integer",
                    "minimum": range[0],
                    "maximum": range[1],
                },
            },
            "required": [score_column_name],
        }

        job_id = self.infer(
            data=data,
            model=model,
            name=name,
            description=description,
            column=column,
            system_prompt=system_prompt,
            output_schema=json_schema,
            job_priority=job_priority,
            stay_attached=False,
        )

        res = self.await_job_completion(job_id)
        if isinstance(data, pl.DataFrame):
            return data.with_columns(
                pl.Series(score_column_name, res[score_column_name])
            )
        elif isinstance(data, pd.DataFrame):
            return data.assign(**{score_column_name: res[score_column_name]})
        else:
            return res


class Rank(BaseSutroClient):
    def rank(
        self,
        model: ModelOptions = "gemma-3-12b-it",
        job_priority: int = 0,
        name: Union[str, List[str]] = None,
        description: Union[str, List[str]] = None,
        # function-specific parameters
        data: Union[
            List[List], pd.DataFrame, pl.DataFrame, str
        ] = None,  # data is always required, but this method accepts a list of lists as well
        option_labels: List[str] = None,
        criteria: Union[str, List[str]] = None,
        ranking_column_name: str = "ranking",
        run_elo: bool = True,
    ):
        """
        A simple invocation of a LLM-as-a-judge ranking (pairwise comparison) function, accepting multiple options to rank.
        Accepts a list of lists, a pandas or polars dataframe, as well as option labels to use for the ranking.

        If using a lists of lists, the option labels should correspond to the labels you would like to use for the ranking, in the same order as the lists.
        If using a pandas or polars dataframe, the option labels should correspond to the column names of the dataframe to use for the ranking.

        Returns a list of lists of rankings ordered from best to worst, corresponding to the option labels.
        If using a pandas or polars dataframe, the rankings will be returned as a column in the original dataframe.
        """

        if isinstance(criteria, str):
            criteria = [criteria]

        system_prompt = f"""You are a judge. Your job is to rank the options presented to you according to the following criteria:
{", ".join(criteria)}
The option labels are: {", ".join(option_labels)}
Return a ranking of the options as an ordered list of the labels from best to worst, and nothing else."""

        json_schema = {
            "type": "object",
            "properties": {
                f"{ranking_column_name}": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [ranking_column_name],
        }

        if isinstance(data, list):
            # create a polars dataframe from the list of lists
            data = pl.DataFrame(data, schema=option_labels)
        elif isinstance(data, pd.DataFrame):
            # convert to polars dataframe
            data = data.from_pandas(data)

        exprs = []  # because the option literals are the same as the same as the column names we don't use the built-in column concatenation helper function
        for _, label in enumerate(option_labels):
            exprs.append(pl.lit(label + ":"))
            exprs.append(pl.col(label))

        data = data.select(
            pl.concat_str(exprs, separator=" ", ignore_nulls=False).alias(
                "options_with_labels"
            )
        )

        job_id = self.infer(
            data=data,
            column="options_with_labels",
            model=model,
            name=name,
            description=description,
            system_prompt=system_prompt,
            output_schema=json_schema,
            job_priority=job_priority,
            stay_attached=False,
        )

        res = self.await_job_completion(job_id, output_column=ranking_column_name)

        # This doesn't work when do as a single step for some reason
        res = (
            res.with_columns(
                pl.col(ranking_column_name).str.json_decode().alias("_decoded")
            )
            .with_columns(
                pl.col("_decoded")
                .struct.field(ranking_column_name)
                .alias(ranking_column_name)
            )
            .drop("_decoded")
        )

        if run_elo:
            elo_ratings = self.elo(data=res, column=ranking_column_name)
            print(elo_ratings[["elo", "wins", "losses", "matches"]].to_markdown())

        if isinstance(data, pl.DataFrame):
            return data.with_columns(
                pl.Series(ranking_column_name, res[ranking_column_name])
            )
        elif isinstance(data, pd.DataFrame):
            return data.assign(**{ranking_column_name: res[ranking_column_name]})
        else:
            return res

    @staticmethod
    def elo(
        data: Union[List, pd.DataFrame, pl.DataFrame] = None,
        column: Union[str, List[str]] = None,
        laplace: float = 0.5,
        max_iter: int = 1000,
        tol: float = 1e-8,
        elo_mean: float = 1500.0,
    ):
        """
        Accepts ordered ranking outputs produced by the rank method, and produces an Elo rating for each label option.
        """

        if isinstance(data, pl.DataFrame):
            if column is None:
                raise ValueError("column is required when using a polars dataframe")
            rankings = data.select(pl.col(column)).to_series().to_list()
        elif isinstance(data, pd.DataFrame):
            if column is None:
                raise ValueError("column is required when using a pandas dataframe")
            rankings = data.loc[:, [column]].tolist()
        else:
            rankings = data

        """ 
        Convert ballots of ordered rankings into pairwise counts, then run Bradley–Terry MM.

        rankings:
        - Strict order: ["B", "A", "C"] means B>A, B>C, A>C
        - With ties: ["B", ("A","C"), "D"] means B > A=C > D
                        (A and C tie once on that ballot)

        Other params are passed to the underlying BT solver.
        """
        # --- build (winner, loser) counts and tie counts from rankings ---
        pair_counts = defaultdict(float)
        tie_counts = defaultdict(float)  # unordered keys (min(name), max(name))

        def as_group(x):
            # allow tuple/list/set to denote a tie group; strings remain atomic
            if isinstance(x, (list, tuple, set)) and not isinstance(x, (str, bytes)):
                return list(x)
            return [x]

        for ballot in rankings:
            # normalize ballot into list of groups, each group a list of items tied at that rank
            groups = [as_group(g) for g in ballot if g is not None]

            # wins across groups: every item in an earlier (better) group beats every item in any later group
            for gi in range(len(groups)):
                for gj in range(gi + 1, len(groups)):
                    for w in groups[gi]:
                        for l in groups[gj]:
                            if w != l:
                                pair_counts[(str(w), str(l))] += 1.0

            # ties within a group: count one tie per unordered pair inside that group
            for g in groups:
                if len(g) >= 2:
                    for a, b in combinations(g, 2):
                        a, b = str(a), str(b)
                        key = (a, b) if a < b else (b, a)
                        if a != b:
                            tie_counts[key] += 1.0

        pair_counts = dict(pair_counts)
        ties = dict(tie_counts) if tie_counts else None

        """
        pair_counts: { (winner, loser): wins } for all observed directed pairs.
        ties: optional { (a, b): tie_count } counted once per unordered pair (a,b).
            If provided, each tie contributes 0.5 win to both directions.
        laplace: additive smoothing to each *directed* count (prevents zeros).
        """

        # ---- Build model list ----
        models = sorted(set([k[0] for k in pair_counts] + [k[1] for k in pair_counts]))
        m = len(models)
        idx = {name: i for i, name in enumerate(models)}

        # ---- Build directed wins matrix W[i,j] = times i beat j ----
        W = np.zeros((m, m), dtype=float)
        for (w, l), c in pair_counts.items():
            if w == l:
                continue
            W[idx[w], idx[l]] += float(c)

        # ---- Optional ties: add 0.5 to both directions for each tie ----
        if ties:
            for (a, b), t in ties.items():
                if a == b:
                    continue
                i, j = idx[a], idx[b]
                W[i, j] += 0.5 * t
                W[j, i] += 0.5 * t

        # ---- Laplace smoothing on directed edges (excluding diagonal) ----
        if laplace and laplace > 0:
            W += laplace
            np.fill_diagonal(W, 0.0)

        # Unordered totals N_ij = W_ij + W_ji
        N = W + W.T
        np.fill_diagonal(N, 0.0)

        # Guard: drop models with zero matches
        active = N.sum(axis=1) > 0
        if not np.all(active):
            keep = np.where(active)[0]
            models = [models[i] for i in keep]
            idx = {name: i for i, name in enumerate(models)}
            W = W[np.ix_(keep, keep)]
            N = N[np.ix_(keep, keep)]
            m = len(models)

        # ---- Bradley–Terry via MM updates (Hunter 2004) ----
        s = np.ones(m, dtype=float)  # abilities (positive)
        for _ in range(max_iter):
            s_old = s.copy()
            w_i = W.sum(axis=1)  # total (smoothed) wins per model
            # denom_i = sum_j N_ij / (s_i + s_j)
            denom = (N / (s.reshape(-1, 1) + s.reshape(1, -1) + 1e-12)).sum(axis=1)
            upd = denom > 0
            s[upd] = w_i[upd] / denom[upd]
            # normalize to keep scale stable (geometric mean = 1)
            s /= np.prod(s) ** (1.0 / m)
            if np.max(np.abs(np.log(s + 1e-12) - np.log(s_old + 1e-12))) < tol:
                break

        # ---- Convert to beta and Elo-like ratings ----
        beta = np.log(s + 1e-12)
        elo = (400.0 / math.log(10.0)) * beta
        elo = elo - np.mean(elo) + elo_mean  # center

        # ---- Summaries and expected probabilities ----
        wins = W.sum(axis=1)
        losses = W.sum(axis=0)
        matches = N.sum(axis=1)  # unordered total vs all opponents

        ratings = pd.DataFrame(
            {
                "ability": s,
                "beta": beta,
                "elo": elo,
                "wins": wins,
                "losses": losses,
                "matches": matches,
            },
            index=models,
        ).sort_values("elo", ascending=False)

        P = s.reshape(-1, 1) / (s.reshape(-1, 1) + s.reshape(1, -1))
        np.fill_diagonal(P, np.nan)
        p_matrix = pd.DataFrame(P, index=models, columns=models)

        return ratings


class EvalTemplates(Score, Rank):
    pass
