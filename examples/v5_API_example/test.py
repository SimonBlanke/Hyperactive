finetuned_opt = OptPipe([("first", FooOpt())("second", BarOpt(params))], more_params)

column_transformer = BetterColumnTransformer(
    [
        {"name": "num", "transformer": StandardScaler(), "columns": ["age", "income"]},
        {"name": "cat", "transformer": OneHotEncoder(), "columns": ["gender", "city"]},
    ]
)


class MyPipeline(Pipeline):
    def transform(self, data):
        numeric = self.columns(["age", "income"]).apply(StandardScaler())
        categorical = self.columns(["gender", "city"]).apply(OneHotEncoder())
        combined = self.concat(numeric, categorical)
        return combined.then(SVC())


finetuned_opt = OptPipe(more_params)
finetuned_opt.add_step(RandomOpt(), fraction=0.3)
finetuned_opt.add_step(fraction=0.4)
finetuned_opt.add_step(fraction=0.3)

finetuned_opt


class OptPipe:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def set_params(self):
        # einzige Möglichkeit um a b c zu ändern


