# Literature the curve-resolution band draws on

What each paper contributed to `resolve_band`, and which papers describe a route that was
looked at and not taken. The decisions themselves are in
[`decisions/0003`](decisions/0003-feasible-band-not-posterior.md) and
[`decisions/0007`](decisions/0007-credible-band-posterior-for-the-identified-part.md).

This file is versioned with the code and not part of the published site. DOIs and titles
were checked against Crossref on 2026-11-19. "Read" says how far: *full* text, *part* of
the text, or *abstract* only.

## Used

| DOI | Title | What we took from it | Read |
| --- | --- | --- | --- |
| [10.3982/ECTA16773](https://doi.org/10.3982/ECTA16773) | Robust Bayesian Inference for Set-Identified Models (Giacomini & Kitagawa, 2021) | The framework: a posterior only for what the data identify, no single prior on the rest, and a credible region for the whole identified set. | abstract |
| [10.3982/ECTA8360](https://doi.org/10.3982/ECTA8360) | Bayesian and Frequentist Inference in Partially Identified Models (Moon & Schorfheide, 2012) | Why a posterior under one prior is too narrow here: its credible sets lie inside the identified set. | abstract |
| [10.1017/S0266466698144043](https://doi.org/10.1017/S0266466698144043) | Revising Beliefs in Nonidentified Models (Poirier, 1998) | For the unidentified part the data do not update the prior, so its posterior is the prior. | abstract |
| [10.1080/00401706.1971.10488823](https://doi.org/10.1080/00401706.1971.10488823) | Self Modeling Curve Resolution (Lawton & Sylvestre, 1971) | The problem itself: non-negativity bounds a set of solutions, not one. | abstract |
| [10.1002/cem.654](https://doi.org/10.1002/cem.654) | Calculation of maximum and minimum band boundaries of feasible solutions for species profiles obtained by multivariate curve resolution (Tauler, 2001) | The band as the minimum and maximum of each profile over all feasible solutions. | abstract |
| [10.1016/j.aca.2016.01.011](https://doi.org/10.1016/j.aca.2016.01.011) | A review of recent methods for the determination of ranges of feasible solutions resulting from soft modelling analyses of multivariate data (Golshan et al., 2016) | The re-mixing matrix `T` on the SVD patterns as the parametrisation of the split, and which constraints narrow it. | abstract |
| [10.1214/aos/1056562461](https://doi.org/10.1214/aos/1056562461) | Slice sampling (Neal, 2003) | The shrinkage step that keeps the hit-and-run walk moving when the region is not convex. | abstract |
| [10.1002/cem.3317](https://doi.org/10.1002/cem.3317) | N-BANDS: A new algorithm for estimating the extension of feasible bands in multivariate curve resolution of multicomponent systems in the presence of noise and rotational ambiguity (Olivieri & Tauler, 2020) | Noise widens the feasible band and has to be part of it, including noise that is not one level everywhere. | abstract |
| [10.1016/j.chemolab.2025.105565](https://doi.org/10.1016/j.chemolab.2025.105565) | Sampling-based computation of the sets of feasible solutions and feasible bands for noisy data (Sawall et al., 2025) | Under noise, exact non-negativity leaves no feasible solution, so small negative values must be tolerated. Its column-wise sampler works for any number of species: the candidate to replace hit-and-run beyond three species. | part |
| no DOI; [preprint, 2026](https://www.numerik.mathematik.uni-rostock.de/storages/uni-rostock/Alle_MNF/Mathe_Numerische_Mathematik/Papers/2026AFS4noisydata.pdf) | Handling experimental data in FACPACK: Noise effects on duality relations and algorithmic parameter selection (Sawall et al.) | The tolerance for negative values should follow from the noise standard deviation (they use 2.5 σ). We derive it from the credibility level and the posterior standard deviation of each value. Their threshold needed a fourfold manual increase on one data set, which is the reason to derive it. | part |

## Looked at, not followed

| DOI | Title | Why not | Read |
| --- | --- | --- | --- |
| [10.1109/TSP.2006.880310](https://doi.org/10.1109/TSP.2006.880310) | Separation of Non-Negative Mixture of Non-Negative Sources Using a Bayesian Approach and MCMC Sampling (Moussaoui et al., 2006) | One prior (Gamma) on sources and mixing coefficients identifies the split. Our bench measured 3–66 % coverage for every prior of this kind. | abstract |
| [10.1109/TSP.2009.2025797](https://doi.org/10.1109/TSP.2009.2025797) | Joint Bayesian Endmember Extraction and Linear Unmixing for Hyperspectral Imagery (Dobigeon et al., 2009) | Same route: positivity, sum-to-one and a prior on a low-dimensional simplex. | abstract |
| [10.1214/25-BA1544](https://doi.org/10.1214/25-BA1544) | Efficiently Resolving Rotational Ambiguity in Bayesian Matrix Sampling with Matching (Poworoznek et al., 2025) | Samples an unidentified factor model and aligns the draws afterwards. We fix the gauge before sampling, which gives one mode and no alignment step. | abstract |
| [10.1016/j.chemolab.2010.05.020](https://doi.org/10.1016/j.chemolab.2010.05.020) | MCR-BANDS: A user friendly MATLAB program for the evaluation of rotation ambiguities in Multivariate Curve Resolution (Jaumot & Tauler, 2010) | Finds the band's extremes by constrained optimisation, one profile at a time. Kept as an alternative to the random walk. | abstract |
| [10.1002/cem.2498](https://doi.org/10.1002/cem.2498) | A fast polygon inflation algorithm to compute the area of feasible solutions for three-component systems. I: concepts and applications (Sawall et al., 2013) | Computes the feasible set exactly, for three species only. | abstract |
| [10.1137/20M1378971](https://doi.org/10.1137/20M1378971) | A Sampling Algorithm to Compute the Set of Feasible Solutions for NonNegative Matrix Factorization with an Arbitrary Rank (Laursen & Hobolth, 2022) | The noise-free sampler that Sawall et al. (2025) extend. Not needed until there are more than three species. | abstract |
