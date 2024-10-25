import torch
import pandas as pd
import altair as alt
from PositionalEncoding import PositionalEncoding


def example_positional():
    pe = PositionalEncoding(20, 0, 100)
    y = pe.forward(torch.zeros(1, 100, 20))

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 7, 8]
        ]
    )

    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=800)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive()
    )


chart = example_positional()
chart.save("visualisation/positional_encoding_chart.html", "html")
