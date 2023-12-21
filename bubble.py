import math
from utils import Topology, ThermostatException, TOLERANCE, Colour, logger, rsqrt, rlog

from vectors import Vector, LorentzVector


class Bubble(Topology):

    def __init__(self, sigma: float, delta: float, mu_r: float, m_uv: float, m_1: float, m_2: float, p: LorentzVector, espilon_term: int = 0, phase: str = 'real'):
        scale = math.sqrt((m_1**2+m_2**2+p.spatial().squared() +
                          p.t**2)/4.)
        super().__init__(scale)
        self.m_1 = m_1
        self.m_2 = m_2
        self.mu_r = mu_r
        self.m_uv = m_uv
        self.p = p
        self.espilon_term = espilon_term
        self.phase = phase
        self.delta = delta
        self.sigma = sigma

        if abs(self.p.squared()/math.sqrt(self.p.t**2+self.p.spatial().squared())) < TOLERANCE:
            raise ThermostatException('p^2 must not be zero')

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
                        1/math.sqrt(k.squared() + self.m_1**2),
                        1/math.sqrt((k+self.p.spatial()
                                     ).squared() + self.m_2**2),
                    ]
                    wgt = self.integrand(
                        k, integrand_implementation, improved_ltd)
                    final_wgt += jac * inv_ose[0]**multi_channeling_power * \
                        wgt / sum(t**multi_channeling_power for t in inv_ose)
                if multi_channeling is True or multi_channeling == 1:
                    k, jac = self.parameterize(
                        xs, parameterization, self.p.spatial()*-1.)
                    inv_ose = [
                        1/math.sqrt(k.squared() + self.m_1**2),
                        1/math.sqrt((k+self.p.spatial()
                                     ).squared() + self.m_2**2)
                    ]
                    wgt = self.integrand(
                        k, integrand_implementation, improved_ltd)
                    final_wgt += jac * inv_ose[1]**multi_channeling_power * \
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

    def hUV(self, sigma: float, r_sq: float) -> float:
        return ((1./sigma)*math.sqrt(1./math.pi))**3*math.exp(-r_sq/sigma**2)

    def hICT(self, sigma: float, t_sq: float) -> float:
        return 4.*(1./sigma**3)*math.sqrt(1./math.pi)*t_sq*math.exp(-t_sq/sigma**2)

    def NCFF(self, emr_qs: list[LorentzVector]) -> float:
        return 1.

    def NCFFUV(self, emr_qs: list[LorentzVector]) -> float:
        return 1.

    def python_integrand(self, loop_momentum: Vector, improved_ltd: bool = False) -> float:

        normalisation = (2*math.pi)**-3
        ose = [
            math.sqrt(loop_momentum.squared() + self.m_1**2),
            math.sqrt((loop_momentum+self.p.spatial()).squared() + self.m_2**2),
        ]
        k_sq = loop_momentum.squared()

        wgts = {
            'integrand': 0.,
            'UV_CT': 0.,
            'integrated_UV_CT': 0.,
            'threshold_CT': 0.,
            'integrated_threshold_CT': 0.,
        }

        threshold_exists = (
            self.m_1+self.m_2)**2 < self.p.squared() and self.p.t > 0.
        match self.phase:
            case 'real':
                match self.espilon_term:
                    case -2: pass
                    case -1:
                        wgts['integrated_UV_CT'] = 1.
                    case 0:
                        if not improved_ltd:
                            cut_solutions = [
                                LorentzVector(ose[0], loop_momentum.x,
                                              loop_momentum.y, loop_momentum.z),
                                LorentzVector(ose[1]+self.p.t, loop_momentum.x,
                                              loop_momentum.y, loop_momentum.z),
                            ]
                            cut_results = [
                                1/(2*ose[0])
                                / ((cut_solutions[0]+self.p).squared()-self.m_2**2),
                                1/((cut_solutions[1]).squared()-self.m_1**2)
                                / (2*ose[1])]
                            wgts['integrand'] = -normalisation*sum(cut_results)
                        else:
                            shifts = [0., -self.p.t]
                            etas = [[ose[i]+ose[j]
                                    for j in range(1)] for i in range(2)]
                            emr_qs_acg_1 = [
                                LorentzVector(ose[0], loop_momentum.x,
                                              loop_momentum.y, loop_momentum.z),
                                LorentzVector(ose[1], loop_momentum.x,
                                              loop_momentum.y, loop_momentum.z)
                            ]
                            emr_qs_acg_2 = [
                                LorentzVector(-ose[0], loop_momentum.x,
                                              loop_momentum.y, loop_momentum.z),
                                LorentzVector(-ose[1], loop_momentum.x,
                                              loop_momentum.y, loop_momentum.z)
                            ]
                            wgts['integrand'] = normalisation/(2*ose[0])/(2*ose[1])*(
                                self.NCFF(emr_qs_acg_1) /
                                (etas[1][0]-shifts[0]+shifts[1])
                                + self.NCFF(emr_qs_acg_2) /
                                (etas[1][0]+shifts[0]-shifts[1])
                            )
                        uv_ose = math.sqrt(
                            loop_momentum.squared() + self.m_uv**2)
                        # Prolly wrong and needs refining for proper cFF UV with num
                        UV_emr_qs = [LorentzVector(uv_ose, loop_momentum.x,
                                                   loop_momentum.y, loop_momentum.z),]*3
                        wgts['UV_CT'] = -normalisation * \
                            self.NCFFUV(UV_emr_qs)/(4*(uv_ose)**3)
                        wgts['integrated_UV_CT'] = - normalisation * (math.pi / 2.) * \
                            math.log(self.m_uv**2/self.mu_r**2) * \
                            self.hUV(self.sigma*self.scale, k_sq)
                        # You can test normalisation of h with the implementation below
                        # wgts['integrated_UV_CT'] = self.h(
                        #     self.sigma*self.scale, k_sq)

                        if threshold_exists:
                            # Now take care of the threshold CT
                            r = math.sqrt(k_sq)
                            t_star = self.solve_radius_rescaling(loop_momentum)
                            r_star = t_star * r
                            ose_star = [
                                math.sqrt(
                                    t_star**2*loop_momentum.squared() + self.m_1**2),
                                math.sqrt(
                                    (loop_momentum*t_star+self.p.spatial()).squared() + self.m_2**2),
                            ]
                            # print("TEST", ose_star[0]+ose_star[1]-self.p.t)
                            if abs(r-r_star)/r_star < self.delta:
                                emr_qs_acg_1 = [
                                    LorentzVector(ose_star[0], t_star*loop_momentum.x,
                                                  t_star*loop_momentum.y, t_star*loop_momentum.z),
                                    LorentzVector(ose_star[1], loop_momentum.x,
                                                  t_star*loop_momentum.y, t_star*loop_momentum.z)
                                ]
                                wgts['threshold_CT'] = - normalisation / (2*ose_star[0]) / (2*ose_star[1]) * (r_star/r)**3 * (
                                    (self.NCFF(emr_qs_acg_1) / (
                                        (t_star**2*k_sq/ose_star[0])+(
                                            (t_star*self.p.spatial().dot(loop_momentum)+t_star**2*k_sq)/ose_star[1])
                                    )) *
                                    (1. / (1-t_star)
                                     )
                                )
                            # print(wgts['threshold_CT'])
                            # print(wgts['integrand'])
                            # print(wgts['integrand']/wgts['threshold_CT'])
                    case _:
                        raise ThermostatException(
                            f'Epsilon term {self.espilon_term} not implemented.')
            case 'imag':
                match self.espilon_term:
                    case -2: pass
                    case -1: pass
                    case 0:
                        if threshold_exists:
                            r = math.sqrt(k_sq)
                            t_star = self.solve_radius_rescaling(loop_momentum)
                            r_star = t_star * r
                            # print(r, r_star, t_star)
                            ose_star = [
                                math.sqrt(
                                    t_star**2*loop_momentum.squared() + self.m_1**2),
                                math.sqrt(
                                    (loop_momentum*t_star+self.p.spatial()).squared() + self.m_2**2),
                            ]
                            emr_qs_acg_1 = [
                                LorentzVector(ose_star[0], t_star*loop_momentum.x,
                                              t_star*loop_momentum.y, t_star*loop_momentum.z),
                                LorentzVector(ose_star[1], loop_momentum.x,
                                              t_star*loop_momentum.y, t_star*loop_momentum.z)
                            ]
                            wgts['integrated_threshold_CT'] = normalisation / (2*ose_star[0]) / (2*ose_star[1]) * (r_star/r)**2 * (
                                (self.NCFF(emr_qs_acg_1) / (
                                    (t_star**2*k_sq/ose_star[0])+(
                                        (t_star*self.p.spatial().dot(loop_momentum)+t_star**2*k_sq)/ose_star[1])
                                )) *
                                self.hICT(self.sigma, t_star**2)
                                * (math.pi / 2.)
                            )
                    case _:
                        raise ThermostatException(
                            f'Epsilon term {self.espilon_term} not implemented.')
            case _:
                raise ThermostatException(
                    f'Phase {self.phase} not implemented.')

        # logger.info(f"Weights for k={loop_momentum}: {wgts}")
        # You can test normalisation of h with the implementation below
        # return wgts['integrated_UV_CT']
        return sum(wgts.values())

    def rust_integrand(self, loop_momentum: Vector, improved_ltd: bool = False) -> float:

        if not improved_ltd:
            raise ThermostatException(
                "Bubble rust implementation only support CFF (improved LTD) formalism.")

        return bubble_integrand(self.delta, self.sigma, self.espilon_term, self.phase, self.mu_r, self.m_1, self.m_2,
                                [loop_momentum.x, loop_momentum.y,
                                    loop_momentum.z],
                                [self.p.t, self.p.x, self.p.y, self.p.z])

    # Solution of -Eshift + Sqrt(c + b*x + a*x**2) + Sqrt(f + e*x + a*x**2) == 0
    #        (c*e - e*f + e*Eshift**2 + b*(-c + f + Eshift**2) +
    # -    2*Eshift*math.sqrt((b - e)*(-(c*e) + b*f) + b*e*Eshift**2 +
    # -       a*((c - f)**2 - 2*(c + f)*Eshift**2 + Eshift**4)))/((b - e)**2 - 4*a*Eshift**2)
    def solve_radius_rescaling(self, k: Vector) -> float:
        c = self.m_1**2
        b = 0.
        a = k.squared()
        e = 2*self.p.spatial().dot(k)
        f = self.m_2**2+self.p.spatial().squared()
        Eshift = self.p.t
        discr = ((b - e)*(-(c*e) + b*f) + b*e*Eshift**2 +
                 a*((c - f)**2 - 2*(c + f)*Eshift**2 + Eshift**4))
        if discr > 0.:
            return (c*e - e*f + e*Eshift**2 + b*(-c + f + Eshift**2) -
                    2*Eshift*math.sqrt(discr))/((b - e)**2 - 4*a*Eshift**2)
        else:
            raise ThermostatException(
                "Cannot rescale momentum to approach non-existing threshold.")

    def analytical_result(self) -> float:
        # Omitting overall factor of i
        normalization = math.pi**2 / (2.*math.pi)**4
        p_sq = self.p.squared()
        m_1_sq = self.m_1**2
        m_2_sq = self.m_2**2
        mu_r_sq = self.mu_r**2
        match (abs(self.m_1), abs(self.m_2)):
            case (0., 0.):
                if p_sq < 0.:
                    res = complex(2 - math.log(-p_sq/self.mu_r**2), 0.)
                else:
                    res = complex(
                        2 - math.log(p_sq/self.mu_r**2), math.pi)
            case (0., _):
                res = 2 + math.log(mu_r_sq / m_2_sq) + ((m_2_sq - p_sq) /
                                                        p_sq)*rlog((m_2_sq - p_sq)/m_2_sq, False)
            case (_, 0.):
                res = 2 + math.log(mu_r_sq / m_1_sq) + ((m_1_sq - p_sq) /
                                                        p_sq)*rlog((m_1_sq - p_sq)/m_1_sq, False)
            case (_, _):
                gamma0 = 1 + m_1_sq/p_sq - m_2_sq/p_sq
                gamma1 = m_1_sq/p_sq
                gamma_sqrt = rsqrt(gamma0**2-4*gamma1, p_sq > 0.)
                gamma_plus = 0.5*(gamma0 + gamma_sqrt)
                gamma_minus = 0.5*(gamma0 - gamma_sqrt)
                res = 2 - rlog(p_sq/mu_r_sq, False)
                for s, g in [(+1, gamma_plus), (-1, gamma_minus)]:
                    log_is_plus = (p_sq > 0. and s > 0) or (
                        p_sq < 0. and s < 0)
                    new_term = g*(rlog((g-1), log_is_plus) -
                                  rlog(g, log_is_plus)) - rlog(g-1, log_is_plus)
                    res += new_term

        real = res.real
        imag = res.imag
        match self.phase:
            case 'real':
                match self.espilon_term:
                    case -2: res = 0.
                    case -1: res = 1.
                    case 0: res = real
                    case _: raise ThermostatException(
                        f'Epsilon term {self.espilon_term} not implemented.')
            case 'imag':
                match self.espilon_term:
                    case -2: res = 0.
                    case -1: res = 0.
                    case 0: res = imag
                    case _: raise ThermostatException(
                        f'Epsilon term {self.espilon_term} not implemented.')
            case _: raise ThermostatException(
                f'Phase {self.phase} not implemented.')

        return res * normalization
