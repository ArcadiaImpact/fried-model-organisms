"""mu-decisiveness: elicit a model's relative sentiment over concepts, fit a 1-D Thurstone
utility model, and report a coherence metric panel (headline: mu-decisiveness).

The submodules import lazily where possible so a pure-API user need not have the local-model
stack installed. Import the pieces you need directly, e.g.::

    from mu_decisiveness.panel import compute_panel
    from mu_decisiveness.fit import fit_caseV_mle
"""

__version__ = "0.1.0"
