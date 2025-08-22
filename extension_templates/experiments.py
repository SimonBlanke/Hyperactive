# copyright: hyperactive developers, MIT License (see LICENSE file)
"""Extension template for experiments.

Purpose of this implementation template:
    quick implementation of new estimators following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new estimator:
- make a copy of the template in a suitable location, give it a descriptive name.
- work through all the "todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- do not write to reserved variables: _tags, _tags_dynamic
- you can add more private methods, but do not override BaseEstimator's private methods
    an easy way to be safe is to prefix your methods with "_custom"
- change docstrings for functions and the file
- ensure interface compatibility by hyperactive.utils.check_estimator
- once complete: use as a local library, or contribute to hyperactive via PR

Mandatory methods:
    scoring         - _score(self, params: dict) -> np.float64
    parameter names - _paramnames(self) -> list[str]

Testing - required for automated test framework and check_estimator usage:
    get default parameters for test instance(s) - get_test_params()
"""
# todo: write an informative docstring for the file or module, remove the above
# todo: add an appropriate copyright notice for your estimator
#       estimators contributed should have the copyright notice at the top
#       estimators of your own do not need to have permissive or MIT copyright

# todo: uncomment the following line, enter authors' GitHub IDs
# __author__ = [authorGitHubID, anotherAuthorGitHubID]

from hyperactive.base import BaseExperiment

# todo: add any necessary imports here

# todo: for imports of soft dependencies:
# make sure to fill in the "python_dependencies" tag with the package import name
# import soft dependencies only inside methods of the class, not at the top of the file


class MyExperiment(BaseExperiment):
    """Custom experiment. todo: write docstring.

    todo: describe your custom experiment here

    Parameters
    ----------
    parama : int
        descriptive explanation of parama
    paramb : string, optional (default='default')
        descriptive explanation of paramb
    paramc : boolean, optional (default=MyOtherEstimator(foo=42))
        descriptive explanation of paramc
    and so on

    Examples
    --------
    >>> from somehwere import MyExperiment
    >>> great_example(code)
    >>> multi_line_expressions(
    ...     require_dots_on_new_lines_so_that_expression_continues_properly
    ... )
    """

    # todo: fill in tags - most tags have sensible defaults below
    _tags = {
        # tags and full specifications are available in the tag API reference
        # TO BE ADDED
        #
        # property tags: subtype
        # ----------------------
        #
        "property:randomness": "random",
        # valid values: "random", "deterministic"
        # if "deterministic", two calls of "evaluate" must result in the same value
        #
        "property:higher_or_lower_is_better": "lower",
        # valid values: "higher", "lower", "mixed"
        # whether higher or lower returns of "evaluate" are better
        #
        # --------------
        # packaging info
        # --------------
        #
        # ownership and contribution tags
        # -------------------------------
        #
        # author = author(s) of th estimator
        # an author is anyone with significant contribution to the code at some point
        "authors": ["author1", "author2"],
        # valid values: str or list of str, should be GitHub handles
        # this should follow best scientific contribution practices
        # scope is the code, not the methodology (method is per paper citation)
        # if interfacing a 3rd party estimator, ensure to give credit to the
        # authors of the interfaced estimator
        #
        # maintainer = current maintainer(s) of the estimator
        # per algorithm maintainer role, see governance document
        # this is an "owner" type role, with rights and maintenance duties
        # for 3rd party interfaces, the scope is the class only
        "maintainers": ["maintainer1", "maintainer2"],
        # valid values: str or list of str, should be GitHub handles
        # remove tag if maintained by package core team
        #
        # dependency tags: python version and soft dependencies
        # -----------------------------------------------------
        #
        # python version requirement
        "python_version": None,
        # valid values: str, PEP 440 valid python version specifiers
        # raises exception at construction if local python version is incompatible
        # delete tag if no python version requirement
        #
        # soft dependency requirement
        "python_dependencies": None,
        # valid values: str or list of str, PEP 440 valid package version specifiers
        # raises exception at construction if modules at strings cannot be imported
        # delete tag if no soft dependency requirement
    }

    # todo: add any hyper-parameters and components to constructor
    def __init__(self, parama, paramb="default", paramc=None):
        # todo: write any hyper-parameters to self
        self.parama = parama
        self.paramb = paramb
        self.paramc = paramc
        # IMPORTANT: the self.params should never be overwritten or mutated from now on
        # for handling defaults etc, write to other attributes, e.g., self._parama

        # leave this as is
        super().__init__()

        # todo: optional, parameter checking logic (if applicable) should happen here
        # if writes derived values to self, should *not* overwrite self.parama etc
        # instead, write to self._parama, self._newparam (starting with _)

    # todo: implement this, mandatory
    def _paramnames(self):
        """Return the parameter names of the search.

        Returns
        -------
        list of str, or None
            The parameter names of the search parameters.
            If not known or arbitrary, return None.
        """
        # for every instance, this should return the correct parameter names
        # i.e., the maximal set of keys of the dict expected by _score
        # (if not known or arbitrary, return None)
        return ["score_param1", "score_param2"]

    # todo: implement this, mandatory
    def _evaluate(self, params):
        """Evaluate the parameters.

        Parameters
        ----------
        params : dict with string keys
            Parameters to evaluate.

        Returns
        -------
        float
            The value of the parameters as per evaluation.
        dict
            Additional metadata about the search.
        """
        # params is a dictionary with keys being paramnames or subset thereof
        # IMPORTANT: avoid side effects to params!
        #
        # the method may work if only a subset of the parameters in paramnames is passed
        # but this is not necessary
        value = 42  # must be numpy.float64
        metadata = {"some": "metadata"}  # can be any dict
        return value, metadata

    # todo: implement this for testing purposes!
    #   required to run local automated unit and integration testing of estimator
    #   method should return default parameters, so that a test instance can be created
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for this type of estimator.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        # todo: set the testing parameters for the estimators
        # Testing parameters can be dictionary or list of dictionaries.
        # Testing parameter choice should cover internal cases well.
        #   for "simple" extension, ignore the parameter_set argument.
        paramset1 = {"parama": 0, "paramb": "default", "paramc": None}
        paramset2 = {"parama": 1, "paramb": "foo", "paramc": 42}
        return [paramset1, paramset2]

        # this method can, if required, use:
        #   class properties (e.g., inherited); parent class test case
        #   imported objects such as estimators from sklearn
        # important: all such imports should be *inside get_test_params*, not at the top
        #            since imports are used only at testing time
        #
        # A good parameter set should primarily satisfy two criteria,
        #   1. Chosen set of parameters should have a low testing time,
        #      ideally in the magnitude of few seconds for the entire test suite.
        #       This is vital for the cases where default values result in
        #       "big" models which not only increases test time but also
        #       run into the risk of test workers crashing.
        #   2. There should be a minimum two such parameter sets with different
        #      sets of values to ensure a wide range of code coverage is provided.
        #
        # example 1: specify params as dictionary
        # any number of params can be specified
        # params = {"est": value0, "parama": value1, "paramb": value2}
        #
        # example 2: specify params as list of dictionary
        # note: Only first dictionary will be used by create_test_instance
        # params = [{"est": value1, "parama": value2},
        #           {"est": value3, "parama": value4}]
        #
        # return params

    @classmethod
    def _get_score_params(self):
        """Return settings for testing score/evaluate functions. Used in tests only.

        Returns a list, the i-th element should be valid arguments for
        self.evaluate and self.score, of an instance constructed with
        self.get_test_params()[i].

        Returns
        -------
        list of dict
            The parameters to be used for scoring.
        """
        # dict keys should be same as paramnames return
        # or subset, only if _evaluate allows for subsets of parameters
        score_params1 = {"score_param1": "foo", "score_param2": "bar"}
        score_params2 = {"score_param1": "baz", "score_param2": "qux"}
        return [score_params1, score_params2]
