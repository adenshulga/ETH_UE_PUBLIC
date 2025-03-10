import typing as tp

import numpy as np
import pandas as pd
from joblib import Memory

from src.data_manipulation.custom_dataset_abc import PricePoint, SizedDataset

memory = Memory("data/cache")


def load_csvs_and_concat_dataframes(
    pathes: list[str], common_columns: list[str]
) -> pd.DataFrame:
    df_list: list[pd.DataFrame] = [pd.read_csv(path)[common_columns] for path in pathes]
    return pd.concat(df_list, ignore_index=True)


# mypy somewhy can't infer the type of Dataset[t], therefore i used here type ignore
class ETH_USD_Binance(SizedDataset[PricePoint]):  # type: ignore
    data_pathes: list[str] = [
        f"data/raw/Binance_ETHUSDT_202{i}_minute.csv" for i in range(1, 5)
    ]

    def __init__(self, start_observations: str | None, end_observations: str | None):
        df = load_csvs_and_concat_dataframes(
            self.data_pathes, common_columns=["date", "low", "high"]
        )

        data = self.preprocess_dataframe(df)

        self.start_observations: np.datetime64 = (
            data["date"][0]
            if start_observations is None
            else np.datetime64(start_observations)
        )

        self.end_observations: np.datetime64 = (
            data["date"][-1]
            if end_observations is None
            else np.datetime64(end_observations)
        )

        mask = (self.start_observations <= data["date"]) & (
            data["date"] <= self.end_observations
        )

        self.data = tp.cast(tp.Sequence[PricePoint], data[mask])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> PricePoint:
        return self.data[index]

    @memory.cache
    @staticmethod
    def preprocess_dataframe(
        df_binance_data: pd.DataFrame,
    ) -> np.recarray:
        """
        - Take as price mean of high and low values
        - Fill the blanks
        - Delete first line with link
        - Change column names to lower case
        """
        df_binance_data["mean_price"] = (
            df_binance_data["high"] + df_binance_data["low"]
        ) / 2

        df_binance_data["date"] = pd.to_datetime(df_binance_data["date"])

        df_binance_data.sort_values(by="date", ascending=True, inplace=True)

        result = df_binance_data[["date", "mean_price"]].to_records(
            index=False, column_dtypes={"date": "datetime64[m]", "mean_price": "float"}
        )

        return result
