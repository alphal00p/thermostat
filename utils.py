import logging
from enum import StrEnum
import math
import cmath
from typing import Any, Callable, Iterator
from vectors import Vector, LorentzVector

try:
    from symbolica import Sample
except ImportError:
    pass


class Colour(StrEnum):
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


logging.basicConfig(
    format=f'{Colour.GREEN}%(levelname)s{Colour.END} {Colour.BLUE}%(funcName)s l.%(lineno)d{Colour.END} {Colour.CYAN}t=%(asctime)s.%(msecs)03d{Colour.END} > %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S'
)
logger = logging.getLogger('Triangler')

TOLERANCE: float = 1e-10


def rsqrt(arg: complex, is_plus: bool) -> complex:
    if abs(arg.imag) > 0.:
        return cmath.sqrt(arg)
    else:
        if arg.real > 0.:
            return complex(math.sqrt(arg.real), 0.)
        else:
            if is_plus:
                return complex(math.sqrt(arg.real), 0.)
            else:
                return complex(-math.sqrt(arg.real), 0.)


def rlog(arg: complex, is_plus: bool) -> complex:
    if abs(arg.imag) > 0.:
        return cmath.log(arg)
    else:
        if arg.real > 0.:
            return complex(math.log(arg.real), 0.)
        else:
            if is_plus:
                return complex(math.log(-arg.real), math.pi)
            else:
                return complex(math.log(-arg.real), -math.pi)


class ThermostatException(Exception):
    pass


def chunks(a_list: list[Any], n: int) -> Iterator[list[Any]]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(a_list), n):
        yield a_list[i:i + n]


class SymbolicaSample(object):
    def __init__(self, sample: Sample):
        self.c: list[float] = sample.c
        self.d: list[int] = sample.d


class Topology(object):

    RESCALING: float = 1.0

    def __init__(self, scale: float, *args, **opts):
        self.scale = scale

    def integrand_xspace(self, xs: list[float], parameterization: str, integrand_implementation: str, improved_ltd: bool = False, multi_channeling: bool | int = True) -> float:
        raise NotImplementedError("Abstract method not implemented.")

    def integrand(self, loop_momentum: Vector, integrand_implementation: str, improved_ltd: bool = False) -> float:
        raise NotImplementedError("Abstract method not implemented.")

    def parameterize(self, xs: list[float], parameterisation: str, origin: Vector | None = None) -> tuple[Vector, float]:
        match parameterisation:
            case 'cartesian': return self.cartesian_parameterize(xs, origin)
            case 'spherical': return self.spherical_parameterize(xs, origin)
            case _: raise ThermostatException(f'Parameterisation {parameterisation} not implemented.')

    def cartesian_parameterize(self, xs: list[float], origin: Vector | None = None) -> tuple[Vector, float]:
        return self.cartesian_parameterize_v3(xs, origin)

    def cartesian_parameterize_v1(self, xs: list[float], origin: Vector | None = None) -> tuple[Vector, float]:
        x, y, z = xs
        scale = self.scale * self.RESCALING
        v = Vector(
            (1/(1-x)-1/x),
            (1/(1-y)-1/y),
            (1/(1-z)-1/z)
        )*scale
        if origin is not None:
            v = v + origin
        jac = scale * (1/(1-x)**2+1/x**2)
        jac *= scale * (1/(1-y)**2+1/y**2)
        jac *= scale * (1/(1-z)**2+1/z**2)
        return (v, jac)

    def cartesian_parameterize_v2(self, xs: list[float], origin: Vector | None = None) -> tuple[Vector, float]:
        x, y, z = xs
        scale = self.scale * self.RESCALING
        v = Vector(
            math.tan((x-0.5)*math.pi),
            math.tan((y-0.5)*math.pi),
            math.tan((z-0.5)*math.pi),
        )*scale
        if origin is not None:
            v = v + origin
        jac = scale * math.pi / math.cos((x-0.5)*math.pi)**2
        jac *= scale * math.pi / math.cos((y-0.5)*math.pi)**2
        jac *= scale * math.pi / math.cos((z-0.5)*math.pi)**2
        return (v, jac)

    def cartesian_parameterize_v3(self, xs: list[float], origin: Vector | None = None) -> tuple[Vector, float]:
        x, y, z = xs
        scale = self.scale * self.RESCALING
        v = Vector(
            math.log(x)-math.log(1-x),
            math.log(y)-math.log(1-y),
            math.log(z)-math.log(1-z),
        )*scale
        if origin is not None:
            v = v + origin
        jac = scale * (1 / x + 1 / (1-x))
        jac *= scale * (1 / y + 1 / (1-y))
        jac *= scale * (1 / z + 1 / (1-z))
        return (v, jac)

    def spherical_parameterize(self, xs: list[float], origin: Vector | None = None) -> tuple[Vector, float]:
        rx, costhetax, phix = xs
        scale = self.scale * self.RESCALING
        r = rx / (1 - rx) * scale
        # r = math.log(1. / (1. - rx)) * scale
        costheta = (0.5-costhetax)*2
        sintheta = math.sqrt(1-costheta**2)
        phi = phix * 2 * math.pi
        v = Vector(
            r * sintheta * math.cos(phi),
            r * sintheta * math.sin(phi),
            r * costheta
        )
        if origin is not None:
            v = v + origin
        jac = 2 * (2 * math.pi) * (r**2 * scale / (1-rx)**2)
        # jac = 2 * (2 * math.pi) * (r**2 * scale / (1-rx))
        return (v, jac)
