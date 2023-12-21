import math
from utils import Topology, ThermostatException, TOLERANCE, Colour, logger

from vectors import Vector, LorentzVector
try:
    from numerical_code import ltd_triangle
except:
    pass


class Triangle(Topology):

    def __init__(self, m_2: float, m_1: float, p: LorentzVector, q: LorentzVector):
        scale = math.sqrt((m_1**2+m_2**2+p.spatial().squared() +
                          p.t**2+q.spatial().squared()+q.t**2)/6.)
        super().__init__(scale)
        self.m_1 = m_1
        self.m_2 = m_2
        self.p = p
        self.q = q

        # Only perform sanity checks if in the physical region
        if (self.p+self.q).squared() > 0. or self.p.squared() > 0. or self.q.squared() > 0.:
            if m_1 <= 0.:
                raise ThermostatException('m_1 must be positive.')
            if abs(p.squared()) / m_1 > TOLERANCE:
                raise ThermostatException('p must be on-shell.')
            if abs(q.squared()) / m_1 > TOLERANCE:
                raise ThermostatException('q must be on-shell.')
            if abs((p+q).squared()-m_1**2)/m_1**2 > TOLERANCE:
                raise ThermostatException('p+q must be on-shell.')

    def integrand_xspace(self, xs: list[float], parameterization: str, integrand_implementation: str, improved_ltd: bool = False, multi_channeling: bool | int = True) -> float:
        try:
            if multi_channeling is False:
                k, jac = self.parameterize(xs, parameterization)
                wgt = self.integrand(k, integrand_implementation, improved_ltd)
                final_wgt = wgt * jac
            else:
                final_wgt = 0.
                multi_channeling_power = 3
                if multi_channeling is True or multi_channeling == 0:
                    k, jac = self.parameterize(
                        xs, parameterization, Vector(0., 0., 0.))
                    inv_ose = [
                        1/math.sqrt(k.squared() + self.m_2**2),
                        1/math.sqrt((k-self.q.spatial()
                                     ).squared() + self.m_2**2),
                        1/math.sqrt((k+self.p.spatial()
                                     ).squared() + self.m_2**2),
                    ]
                    wgt = self.integrand(
                        k, integrand_implementation, improved_ltd)
                    final_wgt += jac * inv_ose[0]**multi_channeling_power * \
                        wgt / sum(t**multi_channeling_power for t in inv_ose)
                if multi_channeling is True or multi_channeling == 1:
                    k, jac = self.parameterize(
                        xs, parameterization, self.q.spatial())
                    inv_ose = [
                        1/math.sqrt(k.squared() + self.m_2**2),
                        1/math.sqrt((k-self.q.spatial()
                                     ).squared() + self.m_2**2),
                        1/math.sqrt((k+self.p.spatial()
                                     ).squared() + self.m_2**2),
                    ]
                    wgt = self.integrand(
                        k, integrand_implementation, improved_ltd)
                    final_wgt += jac * inv_ose[1]**multi_channeling_power * \
                        wgt / sum(t**multi_channeling_power for t in inv_ose)
                if multi_channeling is True or multi_channeling == 2:
                    k, jac = self.parameterize(
                        xs, parameterization, self.p.spatial()*-1.)
                    inv_ose = [
                        1/math.sqrt(k.squared() + self.m_2**2),
                        1/math.sqrt((k-self.q.spatial()
                                     ).squared() + self.m_2**2),
                        1/math.sqrt((k+self.p.spatial()
                                     ).squared() + self.m_2**2),
                    ]
                    wgt = self.integrand(
                        k, integrand_implementation, improved_ltd)
                    final_wgt += jac * inv_ose[2]**multi_channeling_power * \
                        wgt / sum(t**multi_channeling_power for t in inv_ose)

            if math.isnan(final_wgt):
                logger.debug(
                    f"Integrand evaluated to NaN at xs = [{Colour.BLUE}{', '.join(f'{xi:+.16e}' for xi in xs)}{Colour.END}]. Setting it to zero")
                final_wgt = 0.
        except ZeroDivisionError:
            logger.debug(
                f"Integrand divided by zero at xs = [{Colour.BLUE}{', '.join(f'{xi:+.16e}' for xi in xs)}{Colour.END}]. Setting it to zero")
            final_wgt = 0.

        return final_wgt

    def integrand(self, loop_momentum: Vector, integrand_implementation: str, improved_ltd: bool = False) -> float:

        try:
            match integrand_implementation:
                case 'python': return self.python_integrand(loop_momentum, improved_ltd)
                case 'rust': return self.rust_integrand(loop_momentum, improved_ltd)
                case _: raise ThermostatException(f'Integrand implementation {integrand_implementation} not implemented.')
        except ZeroDivisionError:
            logger.debug(
                f"Integrand divided by zero for k = [{Colour.BLUE}{', '.join(f'{ki:+.16e}' for ki in loop_momentum.to_list())}{Colour.END}]. Setting it to zero")
            return 0.

    def python_integrand(self, loop_momentum: Vector, improved_ltd: bool = False) -> float:

        # logger.debug(f"loop_momentum={loop_momentum}")
        # logger.debug(f"self.m_2={self.m_2}")
        # logger.debug(f"self.q={self.q}")
        # logger.debug(f"self.p={self.p}")
        ose = [
            math.sqrt(loop_momentum.squared() + self.m_2**2),
            math.sqrt((loop_momentum-self.q.spatial()).squared() + self.m_2**2),
            math.sqrt((loop_momentum+self.p.spatial()).squared() + self.m_2**2),
        ]

        if not improved_ltd:
            # The original LTD expression for the loop triangle diagram (imaginary part omitted)
            cut_solutions = [
                LorentzVector(ose[0], loop_momentum.x,
                              loop_momentum.y, loop_momentum.z),
                LorentzVector(ose[1]+self.q.t, loop_momentum.x,
                              loop_momentum.y, loop_momentum.z),
                LorentzVector(ose[2]-self.p.t, loop_momentum.x,
                              loop_momentum.y, loop_momentum.z),
            ]
            cut_results = [
                1/(2*ose[0])
                / ((cut_solutions[0]-self.q).squared()-self.m_2**2)
                / ((cut_solutions[0]+self.p).squared()-self.m_2**2),
                1/((cut_solutions[1]).squared()-self.m_2**2)
                / (2*ose[1])
                / ((cut_solutions[1]+self.p).squared()-self.m_2**2),
                1/((cut_solutions[2]).squared()-self.m_2**2)
                / ((cut_solutions[2]-self.q).squared()-self.m_2**2)
                / (2*ose[2])
            ]
            itg = (2*math.pi)**-3*sum(cut_results)
            # logger.debug(f"integrand={itg:.16e}")
            return itg
        else:
            shifts = [0., self.q.t, -self.p.t]
            etas = [[ose[i]+ose[j] for j in range(3)] for i in range(3)]
            # The algebraically equivalent LTD expression for the loop triangle diagram with spurious poles removed (imaginary part omitted)
            # raise NotImplementedError('Improved LTD expression not implemented yet.')
            return (2*math.pi)**-3/(2*ose[0])/(2*ose[1])/(2*ose[2])*(
                1/((etas[0][2]-shifts[0]+shifts[2])
                   * (etas[1][2]-shifts[1]+shifts[2]))
                + 1/((etas[0][1]+shifts[0]-shifts[1])
                     * (etas[1][2]-shifts[1]+shifts[2]))
                + 1/((etas[0][1]+shifts[0]-shifts[1])
                     * (etas[0][2]+shifts[0]-shifts[2]))
                + 1/((etas[0][2]+shifts[0]-shifts[2])
                     * (etas[1][2]+shifts[1]-shifts[2]))
                + 1/((etas[0][1]-shifts[0]+shifts[1])
                     * (etas[1][2]+shifts[1]-shifts[2]))
                + 1/((etas[0][1]-shifts[0]+shifts[1])
                     * (etas[0][2]-shifts[0]+shifts[2]))
            )

    def rust_integrand(self, loop_momentum: Vector, improved_ltd: bool = False) -> float:
        if not improved_ltd:
            raise NotImplementedError(
                'Rust integrand is only implemented for the improved LTD version.')
        else:
            return ltd_triangle(self.m_2,
                                [loop_momentum.x, loop_momentum.y, loop_momentum.z],
                                [self.q.t, self.q.x, self.q.y, self.q.z],
                                [self.p.t, self.p.x, self.p.y, self.p.z])

    def analytical_result(self) -> float:
        if self.m_1 > 2 * self.m_2:
            logger.critical(
                'Analytical result not implemented for m_1 > 2 * m_2. Analytical result set to 0.')
            return 0.
        else:
            return 1/(8*math.pi**2)*1/(self.m_1**2)*math.asin(self.m_1 / (2 * self.m_2))**2
