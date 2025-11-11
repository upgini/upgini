import numpy as np
import pandas as pd


class OneHotDecoder:

    def encode(df: pd.DataFrame, category_columns: list[str]) -> pd.DataFrame:
        """
        Encode categorical columns into one-hot encoded columns.
        """
        return pd.get_dummies(df, columns=category_columns, prefix_sep="")

    def decode(df: pd.DataFrame) -> (pd.DataFrame, dict[str, list[str]], dict[str, list[str]]):
        """
        Detect one-hot encoded column groups and collapse each group into a single
        categorical column. For each row, all active bits in the group are
        encoded into a unique category using a bitmask over the group's columns
        (ordered by numeric suffix). Rows with zero active bits are set to NA.

        Returns a new DataFrame with transformed columns.
        """
        one_hot_candidate_groups = OneHotDecoder._group_one_hot_fast(df.columns)
        true_one_hot_groups: dict[str, list[str]] = {}

        # 1) Detect valid one-hot groups (filter candidates by column-level checks)
        for group_name, column_candidates in one_hot_candidate_groups.items():
            group_columns: list[str] = []
            for column in column_candidates:
                value_counts = df[column].value_counts(dropna=False, normalize=True)
                most_frequent_percent = value_counts.iloc[0]
                if most_frequent_percent >= 0.6 and OneHotDecoder._is_one_hot_encoded(df[column]):
                    group_columns.append(column)
            if len(group_columns) > 1:
                true_one_hot_groups[group_name] = group_columns

        # 2) Transform: replace each detected group with one categorical column
        if not true_one_hot_groups:
            return df, {}, {}

        result_df = df.copy()
        pseudo_one_hot_groups: dict[str, list[str]] = {}
        for group_name, group_columns in true_one_hot_groups.items():
            sub = result_df[group_columns].copy()
            for c in group_columns:
                s = sub[c]
                if s.dtype == "object" or s.dtype == "string":
                    s = s.astype(str).str.strip().str.lower()
                    s = s.replace({"true": "1", "false": "0"})
                s = pd.to_numeric(s, errors="coerce")
                sub[c] = s

            # 3) Find pseudo one-hot encoded columns when there are multiple ones in one row
            if any(sub.sum(axis=1) > 1):
                pseudo_one_hot_groups[group_name] = group_columns
                result_df[group_columns] = result_df[group_columns].astype("string")
                continue

            # Coerce values to numeric 0/1 handling common textual forms
            sub = sub.fillna(0.0)
            # Binarize strictly to 0/1
            bin_values = (sub.to_numpy() > 0.5).astype(np.int64)
            # Map single active bit to exact numeric suffix from column name
            row_sums = bin_values.sum(axis=1)
            argmax_idx = bin_values.argmax(axis=1)
            suffix_arr = np.array(
                [int(OneHotDecoder._split_prefix_numeric_suffix(col)[1]) for col in group_columns], dtype=np.int64
            )
            codes = suffix_arr[argmax_idx]
            categorical_series = pd.Series(codes, index=sub.index)
            # Keep only rows with exactly one active bit; else set NA
            categorical_series = categorical_series.where(row_sums == 1, other=pd.NA)
            # Use pandas nullable integer dtype to keep NA with integer codes
            result_df[group_name] = categorical_series.astype("Int64").astype("string")

            # Drop original one-hot columns of the group
            result_df = result_df.drop(columns=group_columns)

        for group_name in pseudo_one_hot_groups:
            del true_one_hot_groups[group_name]

        return result_df, true_one_hot_groups, pseudo_one_hot_groups

    def decode_with_cached_groups(
        df: pd.DataFrame, true_one_hot_groups: dict[str, list[str]], pseudo_one_hot_groups: dict[str, list[str]]
    ) -> pd.DataFrame:
        """
        Decode one-hot encoded columns with cached groups.
        """
        result_df = df.copy()
        # 1. Transform regular one-hot groups back to categorical
        if true_one_hot_groups:
            for group_name, group_columns in true_one_hot_groups.items():
                sub = result_df[group_columns].copy()
                for c in group_columns:
                    s = sub[c]
                    if s.dtype == "object" or s.dtype == "string":
                        s = s.astype(str).str.strip().str.lower()
                        s = s.replace({"true": "1", "false": "0"})
                    s = pd.to_numeric(s, errors="coerce")
                    sub[c] = s
                sub = sub.fillna(0.0)
                bin_values = (sub.to_numpy() > 0.5).astype(np.int64)
                row_sums = bin_values.sum(axis=1)
                argmax_idx = bin_values.argmax(axis=1)
                suffix_arr = np.array(
                    [int(OneHotDecoder._split_prefix_numeric_suffix(col)[1]) for col in group_columns], dtype=np.int64
                )
                codes = suffix_arr[argmax_idx]
                categorical_series = pd.Series(codes, index=sub.index)
                categorical_series = categorical_series.where(row_sums == 1, other=pd.NA)
                result_df[group_name] = categorical_series.astype("Int64").astype("string")
                result_df = result_df.drop(columns=group_columns)
        # 2. Convert pseudo-one-hot features to string
        if pseudo_one_hot_groups:
            for _, group_columns in pseudo_one_hot_groups.items():
                result_df[group_columns] = result_df[group_columns].astype("string")
        return result_df

    @staticmethod
    def _is_ascii_digit(c: str) -> bool:
        return "0" <= c <= "9"

    @staticmethod
    def _split_prefix_numeric_suffix(name: str) -> tuple[str, str] | None:
        """
        Return (prefix, numeric_suffix) if name ends with ASCII digits and isn't all digits.
        Otherwise None.
        """
        if not name or not OneHotDecoder._is_ascii_digit(name[-1]):
            return None
        i = len(name) - 1
        while i >= 0 and OneHotDecoder._is_ascii_digit(name[i]):
            i -= 1
        if i < 0:
            # Entire string is digits -> reject
            return None
        return name[: i + 1], name[i + 1 :]  # prefix, suffix

    @staticmethod
    def _group_one_hot_fast(
        candidates: list[str], min_group_size: int = 2, require_consecutive: bool = True
    ) -> dict[str, list[str]]:
        """
        Group OHE-like columns by (prefix, numeric_suffix).
        - Only keeps groups with size >= min_group_size (default: 2).
        - Each group's columns are sorted by numeric suffix (int).
        Returns: {prefix: [col_names_sorted]}.
        """
        if min_group_size < 2:
            raise ValueError("min_group_size must be >= 2.")

        # 1) Collect by prefix with parsed numeric suffix
        groups: dict[str, list[(int, str)]] = {}
        for s in candidates:
            sp = OneHotDecoder._split_prefix_numeric_suffix(s)
            if sp is None:
                continue
            prefix, sfx = sp
            groups.setdefault(prefix, []).append((int(sfx), s))

        # 2) Filter and finalize
        out: dict[str, list[str]] = {}
        for prefix, pairs in groups.items():
            if len(pairs) < min_group_size:
                continue
            pairs.sort(key=lambda t: t[0])  # sort by numeric suffix
            if require_consecutive:
                suffixes = [num for num, _ in pairs]
                # no duplicates
                if len(suffixes) != len(set(suffixes)):
                    continue
                # strictly consecutive run with step=1
                start = suffixes[0]
                if any(suffixes[i] != start + i for i in range(len(suffixes))):
                    continue
            out[prefix] = [name for _, name in pairs]

        return out

    def _is_one_hot_encoded(series: pd.Series) -> bool:
        try:
            # All rows should be the same type
            if series.apply(lambda x: type(x)).nunique() != 1:
                return False

            # First, handle string representations of True/False
            series_copy = series.copy()
            if series_copy.dtype == "object" or series_copy.dtype == "string":
                # Convert string representations of boolean values to numeric
                series_copy = series_copy.astype(str).str.strip().str.lower()
                series_copy = series_copy.replace({"true": "1", "false": "0"})

            # Column contains only 0 and 1 (as strings or numbers or booleans)
            series_copy = series_copy.astype(float)
            if set(series_copy.unique()) != {0.0, 1.0}:
                return False

            series_copy = series_copy.astype(int)

            # Column doesn't contain any NaN, np.NaN, space, null, etc.
            if not (series_copy.isin([0, 1])).all():
                return False

            vc = series_copy.value_counts()
            # Column should contain both 0 and 1
            if len(vc) != 2:
                return False

            # Minority class is 1
            if vc[1] >= vc[0]:
                return False

            return True
        except ValueError:
            return False
