# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==5.5.0",
#     "duckdb==1.2.1",
#     "marimo",
#     "matplotlib==3.10.1",
#     "numpy==2.2.4",
#     "openai==1.68.0",
#     "polars==1.25.2",
#     "pyarrow==19.0.1",
#     "scikit-learn==1.6.1",
#     "sqlglot==26.11.1",
#     "typer==0.15.2",
# ]
# ///

import marimo

__generated_with = "0.12.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import altair as alt
    return (alt,)


@app.cell
def _():
    import polars as pl
    return (pl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Programming languages over the years""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""This dataset looks at the relative popularity of various programming languages over the years. Our task is to build models that predict the popularity of programming languages into the future.""")
    return


@app.cell
def _(mo, pl):
    df = pl.read_csv("languages.csv", try_parse_dates=True)
    mo.accordion({"Full data": df})
    return (df,)


@app.cell
def _(df, mo, pl):
    languages = mo.ui.multiselect(df.select(pl.all().exclude("Date")).columns)
    return (languages,)


@app.cell
def _(languages, mo):
    mo.md(
        f"""
        In this notebook, we'll start by exploring the data visually, and then we'll try to predict language popularity into the future.

        Let's start by picking some languages: {languages}.
        """
    )
    return


@app.cell
def _(alt, df, languages, mo, pl):
    mo.ui.altair_chart(
        alt.Chart(
            df.select(["Date"] + languages.value).unpivot(
                pl.selectors.numeric(),
                index="Date",
                variable_name="language",
                value_name="percent",
            )
        )
        .mark_line(opacity=0.75)
        .encode(
            x="Date",
            y=alt.Y("percent:Q"),
            color=alt.Color("language:N"),
        )
    ) if languages.value else None
    return


@app.cell
def _(df, mo):
    _df = mo.sql(
        f"""
        SELECT
            Date,
            Python - LAG(Python) OVER (ORDER BY Date) as Popularity_increase
        FROM df;
        """,
        output=False
    )
    return


@app.cell
def _(df, pl):
    popularity_increases = df.with_columns(
        (pl.all().exclude("Date") - pl.all().exclude("Date").shift(1))
    )
    return (popularity_increases,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Predicting language popularity

        Next, we'll train simple autoregressive models on these trends.
        """
    )
    return


@app.cell
def _(languages):
    languages
    return


@app.cell
def _(mo):
    p_slider = mo.ui.slider(1, 5, label="Window size $p$")
    p_slider
    return (p_slider,)


@app.cell
def _(p_slider):
    p = p_slider.value
    return (p,)


@app.cell
def _(mo):
    steps_slider = mo.ui.slider(1, 10, label="Prediction steps")
    steps_slider
    return (steps_slider,)


@app.cell
def _(steps_slider):
    steps = steps_slider.value
    return (steps,)


@app.cell
def _(df, forecast_ar, languages, np, p, plt, steps, train_ar_model):
    for language in languages.value:
        model = train_ar_model(df, language, p)
        forecasts = forecast_ar(
            model, df[language][-p:].to_numpy(), steps=steps
        ).flatten()
        plt.plot(np.concat([df[language].to_numpy(), forecasts]), label=language)

    ax = None
    if languages.value:
        ax = plt.gca()
        years = np.arange(2004, 2024 + steps, 2)
        ax.set_xticks(np.arange(0, 20 + steps, 2))
        ax.set_xticklabels(years, rotation=45)  # Set tick labels to years with 45° rotation
        ax.axvline(20, linestyle="--", color="black", linewidth="1")
        plt.legend()

    ax
    return ax, forecasts, language, model, years


@app.cell
def _(pl):
    import numpy as np
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt


    def create_ar_features(df, language, p=4):
        """
        Create autoregressive features from a time series

        Parameters:
        series: pandas Series or 1D array - the time series data
        p: int - the order of the AR model (number of lags to use)

        Returns:
        X: DataFrame - feature matrix with lagged values
        y: Series - target values
        """
        # Create lagged features
        df = df.select("Date", language)
        df = df.with_columns(
            *[pl.col(language).shift(i).alias(f"lag_{i}") for i in range(1, p + 1)]
        ).drop_nulls()
        X = df.select(pl.exclude("Date", language)).to_numpy()
        y = df.select(language).to_numpy()
        return X, y


    def train_ar_model(df, language, p=4):
        """
        Train an AR(p) model using least squares

        Parameters:
        series: pandas Series or 1D array - the time series data
        p: int - the order of the AR model (number of lags to use)

        Returns:
        model: trained LinearRegression model
        """
        X, y = create_ar_features(df, language, p)
        model = LinearRegression()
        print(f"Fit an AR model with p={p} on {language}")
        model.fit(X, y)

        return model


    def forecast_ar(model, last_p_values, steps=1):
        """
        Generate forecasts from an AR model

        Parameters:
        model: trained LinearRegression model
        last_p_values: array-like - the last p values of the time series
        steps: int - number of steps to forecast ahead

        Returns:
        forecasts: array - forecasted values
        """
        forecasts = []
        # Make sure last_p_values is in the right format (most recent last)
        current_lags = np.array(last_p_values).flatten()

        for _ in range(steps):
            # Reverse the lags to match the expected input format (lag_1, lag_2, ...)
            X_pred = current_lags[::-1].reshape(1, -1)

            # Make a one-step prediction
            prediction = model.predict(X_pred)[0]
            forecasts.append(prediction)

            # Update lags for the next prediction
            current_lags = np.roll(current_lags, 1)
            current_lags[0] = prediction[0]

        return np.array(forecasts)
    return (
        LinearRegression,
        create_ar_features,
        forecast_ar,
        np,
        plt,
        train_ar_model,
    )


@app.cell
def _(df, mo, train_ar_model):
    import typer


    app = typer.Typer()

    @app.command()
    def train(languages: list[str], p: int = 4):
        import pickle
        for language in languages:
            with open(language + ".pkl", "wb") as f:
                pickle.dump(train_ar_model(df, language), f)

    if not mo.running_in_notebook():
        app()
    return app, train, typer


if __name__ == "__main__":
    app.run()
