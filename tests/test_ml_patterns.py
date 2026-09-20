"""The call structure MPFI needs is declared as data, not written as calls.

sklearn hides the steps of a pipeline in a list of tuples and dispatches them
inside `fit`; pandas `.pipe` receives the transform as an argument. A call graph
built from call syntax alone therefore reports feature-engineering functions as
dead and a pipeline as a single flat step.
"""

import textwrap

from pycg_ml.ml_patterns import ml_edges


def edges(source):
    return ml_edges(textwrap.dedent(source), "train")


def test_pipe_argument_is_called_by_the_enclosing_function():
    result = edges("""
        def add_ratio(df):
            return df

        def prepare(df):
            return df.pipe(add_ratio)
    """)

    assert "train.add_ratio" in result["train.prepare"]


def test_chained_pipes_are_all_resolved():
    result = edges("""
        def add_ratio(df):
            return df

        def drop_outliers(df):
            return df

        def prepare(df):
            return df.pipe(add_ratio).pipe(drop_outliers)
    """)

    assert result["train.prepare"] == {"train.add_ratio", "train.drop_outliers"}


def test_pipeline_steps_are_fitted_in_order():
    result = edges("""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        def build():
            return Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression()),
            ])
    """)

    # What Pipeline.fit does at runtime: every step but the last is fit_transformed.
    assert result["sklearn.pipeline.Pipeline.fit"] == {
        "sklearn.preprocessing.StandardScaler.fit_transform",
        "sklearn.linear_model.LogisticRegression.fit",
    }


def test_column_transformer_fits_every_branch():
    result = edges("""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        def build():
            return ColumnTransformer([
                ("num", StandardScaler(), ["a"]),
                ("cat", OneHotEncoder(), ["b"]),
            ])
    """)

    assert result["sklearn.compose.ColumnTransformer.fit"] == {
        "sklearn.preprocessing.StandardScaler.fit_transform",
        "sklearn.preprocessing.OneHotEncoder.fit_transform",
    }


def test_boosting_callback_is_called_by_train():
    result = edges("""
        import lightgbm as lgb

        def custom_eval(preds, data):
            return "err", 0.0, False

        def run(dtrain):
            return lgb.train({}, dtrain, feval=custom_eval)
    """)

    assert "train.custom_eval" in result["lightgbm.train"]


def test_plain_code_gets_no_extra_edges():
    result = edges("""
        def helper(x):
            return x

        def main():
            return helper(1)
    """)

    assert result == {}
