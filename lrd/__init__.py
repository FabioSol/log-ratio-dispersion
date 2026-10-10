"""Log-ratio dispersion: the implementation behind the thesis's verification checks.

    core        D_r, the r-centre, the canonical subgradient, the exact face test
    solver      the penalised solution x*(eps) at index r
    worst_case  exact ground truth for the adversary's inner problem
    prob        softmax, logsumexp, KL, entropy
    samplers    fixed nominals and the degeneracy-aware sampler
    noise_sweep the Chapter 8 label-noise sweep and the exact collapse threshold
                (CLI: python -m lrd.noise_sweep)
    checks      every check in Appendix B, with explicit pass criteria
    paths       where figures and frozen data live
"""
