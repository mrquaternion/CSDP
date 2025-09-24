# carbonpipeline/processor.py
from datetime import datetime
import re
import numpy as np
import pandas as pd

from .constants import VARIABLES_FOR_PREDICTOR
from .processing_utils import PROCESSORS
from ..config import CarbonPipelineConfig


class DataProcessor:
    """Handles data processing operations for climate and environmental data."""
    
    def __init__(self, config: CarbonPipelineConfig):
        self.config = config

    @staticmethod
    def convert_ameriflux_to_era5(df: pd.DataFrame, pred: str) -> np.ndarray:
        """
        Converts AmeriFlux DataFrame columns to ERA5 predictor values.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing AmeriFlux data.
        pred : str
            The predictor variable name for which ERA5 values are required.

        Returns
        -------
        np.array 
            Array of ERA5 predictor values computed from the DataFrame.

        Notes
        -----
        - The columns required for the predictor are determined by VARIABLES_FOR_PREDICTOR[pred].
        - If a processing function is defined in PROCESSORS for the predictor, it is applied row-wise.
        - If no processing function is found, the first relevant column is returned as a NumPy array.
        """
        cols = VARIABLES_FOR_PREDICTOR[pred]
        func = PROCESSORS.get(pred)
        arr = df[cols].to_numpy(dtype=float)

        if func is None:
            return arr[:, 0]
        return func(*[arr[:, i] for i in range(arr.shape[1])])
        
    
    def load_and_filter_dataframe(self, path: str, start: str, end: str) -> pd.DataFrame:
        """Load and filter dataframe based on time range."""

        df = pd.read_csv(path, on_bad_lines='skip') 

        # Datetime conversion 
        df["timestamp"] = df["timestamp"].apply(self._validate_date_format).pipe(pd.to_datetime)         

        # Filter to the hour
        df = df[(df["timestamp"].dt.minute == 0) & (df["timestamp"].dt.second == 0)]

        # Temporal clamp
        start_ts, end_ts = pd.to_datetime(start), pd.to_datetime(end)
        filtered_df = df[df["timestamp"].between(start_ts, end_ts)].copy()  

        return self._find_missing_rows(filtered_df)
    
    def check_data_file_time_range(self, path: str, start: str, end: str):
        df = pd.read_csv(path, on_bad_lines='skip') 
        df["timestamp"] = df["timestamp"].apply(self._validate_date_format).pipe(pd.to_datetime)         
        df = df[(df["timestamp"].dt.minute == 0) & (df["timestamp"].dt.second == 0)]

        min_ts, max_ts = df["timestamp"].min(), df["timestamp"].max()
        start_ts, end_ts = pd.to_datetime(start), pd.to_datetime(end)

        if start_ts < min_ts or end_ts > max_ts:
            msg = (f"The requested interval [{start_ts} -> {end_ts}] "
                   f"is out of bound for the given CSV file [{min_ts} -> {max_ts}].")
            raise ValueError(msg)

    def _validate_date_format(self, ts: str | int) -> any:
        """Validate and format date strings."""
        if isinstance(ts, str):                                      
            try:
                # Attempt to parse with the main format
                datetime.strptime(ts, self.config.DATETIME_FMT)
                return ts
            except ValueError: 
                # If it fails, try to parse the other common format
                try:
                    regex = r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})'
                    Y, M, D, h, m = re.split(regex, str(ts))[1:6]
                    return f"{Y}-{M}-{D} {h}:{m}:00"
                except (ValueError, IndexError):
                     return pd.NaT # Return Not a Time for unparseable dates
        
        # Handle integer timestamps
        try:
            regex = r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})'
            Y, M, D, h, m = re.split(regex, str(ts))[1:6]
            return f"{Y}-{M}-{D} {h}:{m}:00"
        except (ValueError, IndexError):
            return pd.NaT

    def _find_missing_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify and return rows in the DataFrame that contain missing (NaN) values, excluding the 'timestamp' 
        column. Adds 'year', 'month', 'day' and 'time' columns extracted from the 'timestamp' column for each missing row.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with a 'timestamp' column and other columns to check for missing values.

        Returns
        -------
        pd.DataFrame
            DataFrame containing rows with missing values (excluding 'timestamp').
        """
        miss = df[df.drop(columns="timestamp").isnull().any(axis=1)].copy()
        miss.loc[:, "year"] = miss["timestamp"].dt.year
        miss.loc[:, "month"] = miss["timestamp"].dt.month
        miss.loc[:, "day"] = miss["timestamp"].dt.day
        miss.loc[:, "time"] = miss["timestamp"].dt.strftime('%H:%M:%S')
        return miss

    @staticmethod
    def get_missing_groups(df: pd.DataFrame) -> list[tuple]:
        """Get groups of missing data."""
        return [g for g, _ in df.groupby(["year", "month", "day", "time"])]

    @staticmethod
    def get_request_groups(start: pd.Timestamp, end: pd.Timestamp, monthly: bool) -> list[tuple]:
        """
        Generate groups dynamically for ERA5 requests.
        - If monthly=False → hourly/daily ERA5:
            - Full months if possible
            - Full days if possible
            - Otherwise: exact hourly slices
            Returns (year:str, month:str, days:list[str], hours:list[str])
        - If monthly=True → ERA5 monthly means:
            - Full years if possible
            - Full months if possible
            - Otherwise: fallback to days
            Returns (year:str, months:list[str], days:list[str], hours:list[str])
        """
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        groups: list[tuple] = []
        full_hours = [f"{h:02d}:00" for h in range(24)]

        if monthly:
            # ----------- MONTHLY MEANS LOGIC -----------
            # Case 1: full years
            if (start.month, start.day, start.hour) == (1, 1, 0) and (end.month, end.day, end.hour) == (12, 31, 23):
                for year in range(start.year, end.year + 1):
                    months = [f"{m:02d}" for m in range(1, 13)]
                    days = [f"{d:02d}" for d in range(1, 32)]  # tolerated by CDS
                    groups.append((str(year), months, days, full_hours))
                return groups

            # Case 2: full months
            months = pd.period_range(start=start, end=end, freq="M")
            for month in months:
                month_start, month_end = month.start_time, month.end_time
                if start <= month_start <= end:
                    from calendar import monthrange
                    n_days = monthrange(month.year, month.month)[1]
                    days = [f"{d:02d}" for d in range(1, n_days + 1)]
                    groups.append((str(month.year), [f"{month.month:02d}"], days, full_hours))
            return groups

        else:
            # ----------- HOURLY/DAY LOGIC -----------
            months = pd.period_range(start=start, end=end, freq="M")
            for month in months:
                month_start, month_end = month.start_time, month.end_time
                m_start, m_end = max(start, month_start), min(end, month_end)
                if m_start > m_end:
                    continue

                # Case 1: full month
                if m_start.floor("h") == month_start and m_end.floor("h") >= month_end.floor("h"):
                    from calendar import monthrange
                    n_days = monthrange(month.year, month.month)[1]
                    days = [f"{d:02d}" for d in range(1, n_days + 1)]
                    groups.append((str(month.year), f"{month.month:02d}", days, full_hours))
                    continue

                # Case 2: per day
                days_range = pd.date_range(start=m_start.floor("D"), end=m_end.floor("D"), freq="D")
                for d in days_range:
                    y, mo, da = d.year, f"{d.month:02d}", f"{d.day:02d}"
                    h0 = m_start.hour if d == days_range[0] else 0
                    h1 = m_end.hour if d == days_range[-1] else 23
                    h0, h1 = max(0, min(23, h0)), max(0, min(23, h1))
                    if h0 > h1:
                        continue
                    hours = [f"{h:02d}:00" for h in range(h0, h1 + 1)]
                    groups.append((str(y), mo, [da], hours))

            return groups

 
