from hyperactive.opt._adapters._gfo import _BaseGFOadapter


class PowellsMethod(_BaseGFOadapter):

    def _get_gfo_class(self):
        """Get the GFO class to use.

        Returns
        -------
        class
            The GFO class to use. One of the concrete GFO classes
        """
        from gradient_free_optimizers import PowellsMethod

        return PowellsMethod
