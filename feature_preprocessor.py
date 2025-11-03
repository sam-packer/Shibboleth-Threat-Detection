import pandas as pd
import logging


class FeaturePreprocessor:
    def __init__(self):
        self.win_mode = None
        self.iphone_mode = None
        self.device_mem_median = None
        self.column_medians = {}

    def fit(self, df: pd.DataFrame, feature_columns: list):
        """
        Learns the imputation values from the training dataframe.
        """
        logging.info("Fitting FeaturePreprocessor...")

        if "device_memory_gb" in df.columns and "platform" in df.columns:
            windows_df = df[df["platform"].str.contains("Win", case=False, na=False)]
            if not windows_df.empty:
                windows_mode_series = windows_df["device_memory_gb"].mode()
                if not windows_mode_series.empty:
                    self.win_mode = windows_mode_series.iloc[0]

            iphone_df = df[df["platform"].str.contains("iPhone", case=False, na=False)]
            if not iphone_df.empty:
                iphone_mode_series = iphone_df["device_memory_gb"].mode()
                if not iphone_mode_series.empty:
                    self.iphone_mode = iphone_mode_series.iloc[0]

            self.device_mem_median = df["device_memory_gb"].median()

        for col in feature_columns:
            if col in df.columns:
                self.column_medians[col] = df[col].median()

        logging.info("Fit complete.")
        logging.info(f"  > win_mean: {self.win_mode}")
        logging.info(f"  > iphone_mode: {self.iphone_mode}")
        logging.info(f"  > device_mem_median: {self.device_mem_median}")

    def transform_df(self, df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
        """
        Applies the learned transformations to a DataFrame (for training).
        """
        logging.info("Transforming DataFrame...")

        if "device_memory_gb" in df.columns and "platform" in df.columns:
            missing_mask = df["device_memory_gb"].isna()

            if self.win_mode is not None:
                win_mask = missing_mask & df["platform"].str.contains("Win", case=False, na=False)
                df.loc[win_mask, "device_memory_gb"] = self.win_mode

            if self.iphone_mode is not None:
                iphone_mask = missing_mask & df["platform"].str.contains("iPhone", case=False, na=False)
                df.loc[iphone_mask, "device_memory_gb"] = self.iphone_mode

            if self.device_mem_median is not None:
                df["device_memory_gb"].fillna(self.device_mem_median, inplace=True)

        for col in feature_columns:
            if col in df.columns and col in self.column_medians:
                if df[col].isna().any():
                    df[col].fillna(self.column_medians[col], inplace=True)

        return df

    def transform_single(self, features: dict, feature_columns: list) -> dict:
        """
        Applies the learned transformations to a single feature dict (for inference).
        """

        is_missing = features.get("device_memory_gb") is None

        if is_missing and "platform" in features:
            platform = features.get("platform", "") or ""

            if self.win_mode is not None and "Win" in platform:
                features["device_memory_gb"] = self.win_mode
            elif self.iphone_mode is not None and "iPhone" in platform:
                features["device_memory_gb"] = self.iphone_mode
            elif self.device_mem_median is not None:
                features["device_memory_gb"] = self.device_mem_median

        for col in feature_columns:
            if features.get(col) is None:
                if col in self.column_medians:
                    features[col] = self.column_medians[col]
                else:
                    features[col] = 0.0

        return features