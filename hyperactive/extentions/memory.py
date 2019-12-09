# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

class Memory:
    def __init__(self):
        current_path = os.path.realpath(__file__)
        meta_learn_path, _ = current_path.rsplit("/", 1)

        self.meta_data_path = meta_learn_path + "/meta_data/"

    def _get_opt_meta_data(self, _cand_, X, y):
        results_dict = {}
        para_list = []
        score_list = []

        for key in _cand_._space_.memory.keys():
            pos = np.fromstring(key, dtype=int)
            para = _cand_._space_.pos2para(pos)
            score = _cand_._space_.memory[key]

            if score != 0:
                para_list.append(para)
                score_list.append(score)

        results_dict["params"] = para_list
        results_dict["mean_test_score"] = score_list

        return results_dict

    def collect(self, X, y, _cand_):
        results_dict = self._get_opt_meta_data(_cand_, X, y)

        para_pd = pd.DataFrame(results_dict["params"])
        md_model = para_pd.reindex(sorted(para_pd.columns), axis=1)

        metric_pd = pd.DataFrame(
            results_dict["mean_test_score"], columns=["mean_test_score"]
        )

        md_model = pd.concat([para_pd, metric_pd], axis=1, ignore_index=False)

        return md_model
        
    def _get_hash(self, object):
        return hashlib.sha1(object).hexdigest()

    def _get_func_str(self, func):
        return inspect.getsource(func)
        
    def _get_file_path(self, X_train, y_train, model_func):
        func_str = self._get_func_str(model_func)
        feature_hash = self._get_hash(X_train)
        label_hash = self._get_hash(y_train)

        self.func_path = self._get_hash(func_str.encode("utf-8")) + "/"

        directory = self.meta_data_path + self.func_path
        if not os.path.exists(directory):
            os.makedirs(directory)

        return directory + (
            "metadata"
            + "__feature_hash="
            + feature_hash
            + "__label_hash="
            + label_hash
            + "__.csv"
        )
