# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==5.5.0",
#     "marimo",
#     "matplotlib==3.10.1",
#     "pandas==2.2.3",
#     "scikit-learn==1.6.1",
# ]
# ///

import marimo

__generated_with = "0.12.10"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Embedding Visualizer""")
    return


@app.cell
def _():
    import sklearn
    import sklearn.datasets
    import sklearn.manifold

    raw_digits, raw_labels = sklearn.datasets.load_digits(return_X_y=True)
    return raw_digits, raw_labels, sklearn


@app.cell
def _(pd, raw_digits, raw_labels, sklearn):
    X_embedded = sklearn.decomposition.PCA(
        n_components=2, whiten=True
    ).fit_transform(raw_digits)

    embedding = pd.DataFrame(
        {"x": X_embedded[:, 0], "y": X_embedded[:, 1], "digit": raw_labels}
    ).reset_index()
    return X_embedded, embedding


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        f"""
        Here's a PCA **embedding of numerical digits**: each point represents a 
        digit, with similar digits close to each other. The data is from the UCI 
        ML handwritten digits dataset.

        This notebook will automatically drill down into points you **select with 
        your mouse**; try it!
        """
    )
    return


@app.cell
def _(embedding, mo, scatter):
    chart = mo.ui.altair_chart(scatter(embedding))
    chart
    return (chart,)


@app.cell
def _(chart):
    chart.value
    return


@app.cell
def _(chart, mo):
    table = mo.ui.table(chart.value)
    return (table,)


@app.cell(hide_code=True)
def _(chart, mo, raw_digits, table):
    # show 10 images: either the first 10 from the selection, or the first ten
    # selected in the table
    mo.stop(not len(chart.value))


    def show_images(indices, max_images=10):
        import matplotlib.pyplot as plt

        indices = indices[:max_images]
        images = raw_digits.reshape((-1, 8, 8))[indices]
        fig, axes = plt.subplots(1, len(indices))
        fig.set_size_inches(12.5, 1.5)
        if len(indices) > 1:
            for im, ax in zip(images, axes.flat):
                ax.imshow(im, cmap="gray")
                ax.set_yticks([])
                ax.set_xticks([])
        else:
            axes.imshow(images[0], cmap="gray")
            axes.set_yticks([])
            axes.set_xticks([])
        plt.tight_layout()
        return fig


    selected_images = (
        show_images(list(chart.value["index"]))
        if not len(table.value)
        else show_images(list(table.value["index"]))
    )

    mo.md(
        f"""
        **Here's a preview of the images you've selected**:

        {mo.as_html(selected_images)}

        Here's all the data you've selected.

        {table}
        """
    )
    return selected_images, show_images


@app.cell
def _(alt):
    def scatter(df):
        return (
            alt.Chart(df)
            .mark_circle()
            .encode(
                x=alt.X("x:Q").scale(domain=(-2.5, 2.5)),
                y=alt.Y("y:Q").scale(domain=(-2.5, 2.5)),
                color=alt.Color("digit:N"),
            )
            .properties(width=500, height=500)
        )
    return (scatter,)


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _():
    import altair as alt
    return (alt,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
