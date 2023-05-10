""""""

from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum, unique

import sympy as sm
import sympy.physics.mechanics as me

from ollie.human import HumanBase
from ollie.skateboard import Skateboard


__all__ = ["Ollie"]


@unique
class OlliePhase(str, Enum):
    Preparation = "preparation"
    PrePop = "pre-pop"
    Pop = "pop"
    UpwardMotion = "upward motion"
    PeakHeight = "peak height"
    DownwardMotion = "downward motion"
    Landing = "landing"


class OlliePhaseBase(ABC):
    def __init__(self) -> None:
        pass

    @classmethod
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def state_variables(self):
        pass

    @property
    @abstractmethod
    def control_variables(self):
        pass

    def _calculate_equations_of_motion(self):
        """"""
        # self._x_s = me.dynamicsymbols("x_s")
        # self._y_s = me.dynamicsymbols("y_s")
        # self._th_s = me.dynamicsymbols("th_s")
        # self._s_1 = me.dynamicsymbols("s_1")
        # self._s_2 = me.dynamicsymbols("s_2")
        # self._x_w = me.dynamicsymbols("x_w")

        # self._t = me.dynamicsymbols._t
        # self._q = sm.Matrix(
        #     [
        #         self._x_s,
        #         self._y_s,
        #         self._th_s,
        #         self._s_1,
        #         self._s_2,
        #         self._x_w,
        #     ]
        # )
        # self._dq = self._q.diff(self._t)
        # self._ddq = self._dq.diff(self._t)

        # self._N = me.ReferenceFrame("N")
        # self._A = me.ReferenceFrame("A")
        # self._B = me.ReferenceFrame("B")
        # self._C = me.ReferenceFrame("C")

        # self._A.orient_axis(self._N, self._N.z, self._th_s)  # Body-fixed frame
        # self._B.orient_axis(self._N, self._N.z, self._th_s - self.skateboard.phi)
        # self._C.orient_axis(self._N, self._N.z, self._th_s + self.skateboard.phi)

        # self._O = me.Point("O")
        # self._WC1_a = self._O.locatenew("contact back wheels", self._x_w * self._N.x)
        # self._W1_a = self._WC1_a.locatenew(
        #     "back wheels", self.skateboard.r_w * self._N.y
        # )
        # self._Tr1_a = self._W1_a.locatenew(
        #     "back trucks at deck", self.skateboard.d_tr * self._A.y
        # )
        # self._mid_a = self._Tr1_a.locatenew(
        #     "middle deck",
        #     sm.S.Half * self.skateboard.l_wb * self._A.x,
        # )
        # self._com_a = self._mid_a.locatenew("centre of mass", -self._d_com_ * self._A.y)
        # self._B0_a = self._mid_a.locatenew("back pocket", -self._l_f / 2 * self._A.x)
        # self._tail_a = self._B0_a.locatenew("tail", -self._l_t * self._B.x)
        # self._bf_a = self._tail_a.locatenew("back foot", self._s_1 * self._B.x)
        # self._ff_a = self._B0_a.locatenew("front foot", self._s_2 * self._A.x)


class PreparationPhase(OlliePhaseBase):
    name = OlliePhase.Preparation.value

    @property
    def state_variables(self):
        raise NotImplementedError

    @property
    def control_variables(self):
        raise NotImplementedError


class PrePopPhase(OlliePhaseBase):
    name = OlliePhase.PrePop.value

    @property
    def state_variables(self):
        raise NotImplementedError

    @property
    def control_variables(self):
        raise NotImplementedError


class PopPhase(OlliePhaseBase):
    name = OlliePhase.Pop.value

    @property
    def state_variables(self):
        raise NotImplementedError

    @property
    def control_variables(self):
        raise NotImplementedError


class UpwardMotionPhase(OlliePhaseBase):
    name = OlliePhase.UpwardMotion.value

    @property
    def state_variables(self):
        raise NotImplementedError

    @property
    def control_variables(self):
        raise NotImplementedError


class PeakHeightPhase(OlliePhaseBase):
    name = OlliePhase.PeakHeight.value

    @property
    def state_variables(self):
        raise NotImplementedError

    @property
    def control_variables(self):
        raise NotImplementedError


class DownwardMotionPhase(OlliePhaseBase):
    name = OlliePhase.DownwardMotion.value

    @property
    def state_variables(self):
        raise NotImplementedError

    @property
    def control_variables(self):
        raise NotImplementedError


class LandingPhase(OlliePhaseBase):
    name = OlliePhase.Landing.value

    @property
    def state_variables(self):
        raise NotImplementedError

    @property
    def control_variables(self):
        raise NotImplementedError


class Ollie:
    def __init__(self, skateboard: Skateboard, human: HumanBase, phases: list[str | OlliePhase | OlliePhaseBase]):
        self.skateboard = skateboard
        self.human = human
        self.phases = phases

        self.frame = me.ReferenceFrame(r"N_{ollie}")

        # Attach feet to skateboard
        self.human.rear_foot.set_pos(
            self.skateboard.deck.back_pocket,
            -self.skateboard.deck.tail_length * self.human.pos_rear_foot * self.skateboard.deck.tail_frame.x,
        )
        self.human.front_foot.set_pos(
            self.skateboard.deck.back_pocket,
            self.skateboard.deck.length * self.human.pos_front_foot * self.skateboard.deck.frame.x,
        )
        self.human.mass_center.set_pos(
            self.human.rear_foot,
            self.human.pos_mass_center_x * self.human.feet_frame.x + self.human.pos_mass_center_y * self.human.feet_frame.y,
        )

    @property
    def skateboard(self) -> Skateboard:
        return self._skateboard

    @skateboard.setter
    def skateboard(self, skateboard: Skateboard) -> None:
        if hasattr(self, "_skateboard"):
            msg = (
                f"Cannot reset skateboard {self.skateboard} to {skateboard} "
                f"once set"
            )
            raise AttributeError(msg)
        if not isinstance(skateboard, Skateboard):
            msg = f"Skateboard must be a `Skateboard`, not a {type(skateboard)}"
            raise TypeError(msg)
        self._skateboard = skateboard

    @property
    def human(self) -> HumanBase:
        """"""
        return self._human

    @human.setter
    def human(self, human: HumanBase) -> None:
        """"""
        if hasattr(self, "_human"):
            msg = f"Cannot reset human {self.human} to {human} once set"
            raise AttributeError(msg)
        if not isinstance(human, HumanBase):
            msg = f"Human must be a `HumanBase`, not a {type(human)}"
            raise TypeError(msg)
        self._human = human

    @property
    def phases(self) -> tuple[OlliePhaseBase]:
        return self._phases

    @phases.setter
    def phases(
        self,
        phases: list[str | OlliePhase | OlliePhaseBase],
    ) -> None:
        if hasattr(self, "_phases"):
            msg = f"Cannot reset phases {self.phases} to {phases} once set"
            raise AttributeError(msg)
        if not isinstance(phases, (list, tuple)):
            msg = (
                f"`phases` must be an ordered iterable (e.g. `list` or `tuple`) "
                f"of phases, not {phases}"
            )
            raise TypeError(msg)
        ollie_phases = []
        for i, phase in enumerate(phases):
            if isinstance(phase, str):
                ollie_phase = self._instantiate_ollie_phase(OlliePhase(phase))
            elif isinstance(phase, OlliePhase):
                ollie_phase = self._instantiate_ollie_phase(phase)
            elif isinstance(phase, OlliePhaseBase):
                ollie_phase = phase
            else:
                msg = (
                    f"Phase {phase} at index {i} must be a `str`, `OlliePhase`, "
                    f"or `OlliePhaseBase`, not {type(phase)}"
                )
                raise TypeError(msg)
            ollie_phases.append(ollie_phase)
        if len(ollie_phases) != len(set(ollie_phases)):
            msg = "Phases must be unique"
            raise ValueError(msg)
        self._phases = tuple(ollie_phases)

    @staticmethod
    def _instantiate_ollie_phase(ollie_phase: OlliePhase) -> OlliePhaseBase:
        if ollie_phase == OlliePhase.Preparation:
            return PreparationPhase()
        elif ollie_phase == OlliePhase.PrePop:
            return PrePopPhase()
        elif ollie_phase == OlliePhase.Pop:
            return PopPhase()
        elif ollie_phase == OlliePhase.UpwardMotion:
            return UpwardMotionPhase()
        elif ollie_phase == OlliePhase.PeakHeight:
            return PeakHeightPhase()
        elif ollie_phase == OlliePhase.DownwardMotion:
            return DownwardMotionPhase()
        elif ollie_phase == OlliePhase.Landing:
            return LandingPhase()
        msg = (
            f"`ollie_phase` {ollie_phase} is not recognised, must be a value of "
            f"`OlliePhase`"
        )
        raise NotImplementedError(msg)
