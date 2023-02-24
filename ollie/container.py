""""""

from __future__ import annotations

from sympy import Expr, Symbol
from sympy.physics.mechanics import dynamicsymbols


class Container:
    """"""

    def __init__(
        self,
        *,
        symbol: Symbol | dynamicsymbols | None = None,
        value: Expr | float | None = None,
        bounds: tuple[float, float] | None = None,
        guess: float | None = None,
    ):
        """"""
        self._symbol = symbol
        self._value = value
        self._bounds = bounds
        self._guess = guess

    @property
    def symbol(self) -> Symbol | dynamicsymbols | None:
        return self._symbol

    @property
    def value(self) -> float | None:
        """"""
        return self._value

    @property
    def bounds(self) -> tuple[float, float] | None:
        return self._bounds

    @bounds.setter
    def bounds(self, bounds) -> None:
        if hasattr(self, "_bounds"):
            msg = f"Cannot reset bounds {self.bounds} to {bounds} once set"
            raise AttributeError(msg)
        if bounds is None:
            self._bounds = None
            return
        if len(bounds) != 2:
            msg = f"Bounds must be an iterable of length 2, not {bounds}"
            raise TypeError(msg)
        if not isinstance(bounds[0], (int, float)):
            msg = f"Lower bound {bounds[0]} must be a number"
            raise TypeError(msg)
        if not isinstance(bounds[1], (int, float)):
            msg = f"Upper bound {bounds[1]} must be a number"
            raise TypeError(msg)
        if bounds[0] > bounds[1]:
            msg = (
                f"Lower bound {bounds[0]} must be less than upper bound " f"{bounds[1]}"
            )
            raise ValueError(msg)
        self._bounds = tuple(float(bounds[0]), float(bounds[1]))

    @property
    def guess(self) -> float | None:
        return self._guess

    @guess.setter
    def guess(self, guess) -> None:
        if hasattr(self, "_guess"):
            msg = f"Cannot reset guess {self.guess} to {guess} once set"
            raise AttributeError(msg)
        if guess is None:
            self._guess = None
            return
        if not isinstance(guess, (int, float)):
            msg = f"Guess {guess} must be a number"
            raise TypeError(msg)
        if guess < self.bounds[0] or guess > self.bounds[1]:
            msg = f"Guess {guess} must be between bounds {self.bounds}"

    def __neg__(self):
        return -self.symbol

    def __add__(self, other):
        return self.symbol + other

    def __radd__(self, other):
        return other + self.symbol

    def __sub__(self, other):
        return self.symbol - other

    def __rsub__(self, other):
        return other - self.symbol

    def __mul__(self, other):
        return self.symbol * other

    def __rmul__(self, other):
        return other * self.symbol

    def __truediv__(self, other):
        return self.symbol / other

    def __rtruediv__(self, other):
        return other / self.symbol

    def __pow__(self, other):
        return self.symbol**other

    def __rpow__(self, other):
        return other**self.symbol

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(symbol={repr(super())}, "
            f"value={self.value}, bounds={self.bounds}, guess={self.guess})"
        )

    __str__ = __repr__
