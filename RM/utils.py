import os

import pandas as pd
import pandas_ta as ta

from global_vars import dir_strat_df

initial_columns = ["Open", "High", "Low", "Close", "Volume"]


# Function that converts 1 minute data to n minute data with the following columns:
# Timestamp, Open, High, Low, Close, Volume
def convert_to_n_minute_data(df_input: pd.DataFrame, n: int) -> pd.DataFrame:
    """Converts 1 minute data to n minute data with the following columns: Timestamp, Open, High, Low, Close, Volume

    Args:
        df_input (pd.DataFrame): Dataframe with columns: Timestamp, Open, High, Low, Close, Volume
        n (int): Aggregation time in minutes

    Returns:
        pd.DataFrame: Dataframe aggregated to n minute data
    """
    df_input["Timestamp"] = pd.to_datetime(df_input["Timestamp"], unit="s")
    df_input = df_input.set_index("Timestamp")
    df_input = df_input.resample(f"{n}T").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})  # type: ignore # noqa: PGH003
    return df_input.dropna()


def get_strat_df(agg_time: int = 360) -> pd.DataFrame:
    """Get strategies dataframe

    Args:
        agg_time (int, optional): Aggregation time in minutes. Defaults to 360.

    Returns:
        pd.DataFrame: _description_
    """
    file_df_strategies = f"df_strats_{agg_time}_min.parquet"

    file_df_strategies = os.path.join(dir_strat_df, file_df_strategies)

    if not os.path.exists(dir_strat_df):
        os.makedirs(dir_strat_df)

    has_computed_strategies = os.path.exists(file_df_strategies)

    if not has_computed_strategies:
        df_strats = pd.read_parquet("btcusd_1-min_data.parquet")
        df_strats = df_strats.dropna()

        df_strats = convert_to_n_minute_data(df_strats, 60 * 6)

        # Add small value to avoid division by zero
        df_strats["Volume"] += 1e-10
        df_strats = df_strats.set_index(pd.DatetimeIndex(df_strats.index), drop=True)

        df_strats.ta.strategy(ta.AllStrategy, verbose=True)
        df_strats.to_parquet(file_df_strategies)
        print(f"Saved df_strategies: {file_df_strategies}")
    else:
        df_strats = pd.read_parquet(file_df_strategies)

    return df_strats
