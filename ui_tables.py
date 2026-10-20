"""Select scalar metrics before constructing display tables."""
import pandas as pd


def metric_table(results, columns):
    return pd.DataFrame.from_dict(
        {name: {column: float(metrics[column]) for column in columns}
         for name, metrics in results.items()},
        orient='index', columns=columns,
    )
