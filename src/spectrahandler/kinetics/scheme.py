"""First-order reaction schemes and their concentration profiles in closed form.

A scheme is a list of steps ``reactant -> product``, each with its own rate constant.
Every step conserves the total concentration, so a scheme is closed:
``dc/dt = K c`` with the columns of ``K`` summing to zero, and
``c(t) = expm(K t) c(0)``.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import expm

__all__ = ["Scheme", "Step", "concentrations", "rate_matrix"]

#: One first-order step, ``(reactant, product)``.
type Step = tuple[str, str]
#: A step as indices into a species tuple.
type IndexStep = tuple[int, int]


@dataclass(frozen=True, init=False)
class Scheme:
    """A closed scheme of first-order steps.

    ``Scheme(steps=[("A", "B"), ("B", "C")])`` is A -> B -> C. ``("A", "B")`` together
    with ``("B", "A")`` is a reversible pair. Every step has its own rate constant;
    rates are always given and returned in the order of ``steps``.

    Attributes:
        steps: The steps, in the order given.
    """

    steps: tuple[Step, ...]

    def __init__(self, steps: Sequence[Step]) -> None:
        """Validate and store the steps.

        Args:
            steps: ``(reactant, product)`` pairs of species names.

        Raises:
            ValueError: If there are no steps, a step is not a pair of two different
                non-empty names, or a step appears twice.
        """
        checked: list[Step] = []
        for step in steps:
            if (
                isinstance(step, str)
                or not isinstance(step, Sequence)
                or len(step) != 2
                or not all(isinstance(n, str) and n for n in step)
            ):
                raise ValueError(f"a step is a (reactant, product) pair of names, got {step!r}")
            if step[0] == step[1]:
                raise ValueError(f"a step must change the species, got {step!r}")
            checked.append((step[0], step[1]))
        if not checked:
            raise ValueError("a scheme needs at least one step")
        if len(set(checked)) != len(checked):
            raise ValueError(f"steps must be unique, got {checked!r}")
        object.__setattr__(self, "steps", tuple(checked))

    @property
    def species(self) -> tuple[str, ...]:
        """Every species named in a step, sorted alphabetically."""
        return tuple(sorted({name for step in self.steps for name in step}))

    @property
    def n_steps(self) -> int:
        """Number of steps, which is the number of rate constants."""
        return len(self.steps)

    def index_steps(self, species: tuple[str, ...]) -> tuple[IndexStep, ...]:
        """The steps as ``(reactant, product)`` indices into ``species``.

        Args:
            species: The dataset's species, in its order.

        Returns:
            One index pair per step, in step order.

        Raises:
            ValueError: If the scheme's species and ``species`` are not the same set.
        """
        if set(species) != set(self.species):
            raise ValueError(
                f"scheme species {self.species} differ from dataset species {tuple(species)}"
            )
        position = {name: i for i, name in enumerate(species)}
        return tuple((position[a], position[b]) for a, b in self.steps)


def rate_matrix(rates: Array, steps: tuple[IndexStep, ...], n_species: int) -> Array:
    """The matrix ``K`` of ``dc/dt = K c``.

    Step ``i`` moves ``rates[i] * c[reactant]`` from reactant to product, so every column
    sums to zero.

    Args:
        rates: Shape ``(n_steps,)``, in 1 / time unit.
        steps: Index pairs from :meth:`Scheme.index_steps`.
        n_species: Number of species.

    Returns:
        Shape ``(n_species, n_species)``.
    """
    reactant = jnp.asarray([a for a, _ in steps])
    product = jnp.asarray([b for _, b in steps])
    k = jnp.zeros((n_species, n_species), dtype=rates.dtype)
    return k.at[reactant, reactant].add(-rates).at[product, reactant].add(rates)


def concentrations(
    time: Array, rates: Array, initial: Array, steps: tuple[IndexStep, ...]
) -> Array:
    """Concentration profiles ``c(t) = expm(K t) c(0)``.

    ``expm`` stays exact for equal and for reversible rates, where an eigendecomposition
    of ``K`` is defective or complex.

    Args:
        time: Shape ``(n_time,)``, in the time unit of ``rates``.
        rates: Shape ``(n_steps,)``.
        initial: Initial concentrations, shape ``(n_species,)``.
        steps: Index pairs from :meth:`Scheme.index_steps`.

    Returns:
        Shape ``(n_time, n_species)``, in the unit of ``initial``.
    """
    k = rate_matrix(rates, steps, initial.shape[0])
    profiles: Array = jax.vmap(lambda t: expm(k * t) @ initial)(time)
    return profiles
