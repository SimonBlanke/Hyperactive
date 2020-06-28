# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .search_process_shortMem import SearchProcessShortMem


class SearchProcessLongMem(SearchProcessShortMem):
    def __init__(self, nth_process, pro_arg, verb, hyperactive):
        super().__init__(nth_process, pro_arg, verb, hyperactive)

