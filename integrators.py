import math
from utils import Topology, Colour, logger, SymbolicaSample
from typing import Any, Callable
import time
import random
import multiprocessing
import copy

from utils import chunks
try:
    import vegas
except:
    pass

try:
    from symbolica import Sample, NumericalIntegrator
except ImportError:
    pass


class IntegrationResult(object):

    def __init__(self,
                 central_value: float, error: float, n_samples: int = 0, elapsed_time: float = 0.,
                 max_wgt: float | None = None,
                 max_wgt_point: list[float] | None = None):
        self.n_samples = n_samples
        self.central_value = central_value
        self.error = error
        self.max_wgt = max_wgt
        self.max_wgt_point = max_wgt_point
        self.elapsed_time = elapsed_time

    def combine_with(self, other):
        """ Combine self statistics with all those of another IntegrationResult object."""
        self.n_samples += other.n_samples
        self.elapsed_time += other.elapsed_time
        self.central_value += other.central_value
        self.error += other.error
        if other.max_wgt is not None:
            if self.max_wgt is None or abs(self.max_wgt) > abs(other.max_wgt):
                self.max_wgt = other.max_wgt
                self.max_wgt_point = other.max_wgt_point

    def normalize(self):
        """ Normalize the statistics."""
        self.central_value /= self.n_samples
        self.error = math.sqrt(
            abs(self.error / self.n_samples - self.central_value**2)/self.n_samples)

    def str_report(self, target: float | None = None) -> str:

        if self.central_value == 0. or self.n_samples == 0:
            return 'No integration result available yet'

        # First printout sample and timing statitics
        report = [
            f'Integration result after {Colour.GREEN}{self.n_samples}{Colour.END} evaluations in {Colour.GREEN}{self.elapsed_time:.2f} CPU-s{Colour.END}']
        if self.elapsed_time > 0.:
            report[-1] += f' {Colour.BLUE}({1.0e6*self.elapsed_time/self.n_samples:.1f} µs / eval){Colour.END}'

        # Also indicate max weight encountered if provided
        if self.max_wgt is not None and self.max_wgt_point is not None:
            report.append(
                f"Max weight encountered = {self.max_wgt:.5e} at xs = [{' '.join(f'{x:.16e}' for x in self.max_wgt_point)}]")

        # Finally return information about current best estimate of the central value
        report.append(
            f'{Colour.GREEN}Central value{Colour.END} : {self.central_value:<+25.16e} +/- {self.error:<12.2e}')

        err_perc = self.error/self.central_value*100
        if err_perc < 1.:
            report[-1] += f' ({Colour.GREEN}{err_perc:.3f}%{Colour.END})'
        else:
            report[-1] += f' ({Colour.RED}{err_perc:.3f}%{Colour.END})'

        # Also indicate distance to target if specified
        if target is not None and target != 0.:
            report.append(
                f'    vs target : {target:<+25.16e} Δ = {self.central_value-target:<+12.2e}')
            diff_perc = (self.central_value-target)/target*100
            if abs(diff_perc) < 1.:
                report[-1] += f' ({Colour.GREEN}{diff_perc:.3f}%{Colour.END}'
            else:
                report[-1] += f' ({Colour.RED}{diff_perc:.3f}%{Colour.END}'
            if abs(diff_perc/err_perc) < 3.:
                report[-1] += f' {Colour.GREEN} = {abs(diff_perc/err_perc):.2f}σ{Colour.END})'
            else:
                report[-1] += f' {Colour.RED} = {abs(diff_perc/err_perc):.2f}σ{Colour.END})'

        # Join all lines and return
        return '\n'.join(f'| > {line}' for line in report)


class Integrator(object):

    def integrate(self, topology: Topology, parameterisation: str, integrand_implementation: str, improved_ltd: bool, target, **opts) -> IntegrationResult:
        raise NotImplementedError("Abstract method not implemented.")


class NaiveIntegrator(Integrator):

    def __init__(self, *args, **opts):
        super().__init__(*args, **opts)

    @staticmethod
    def naive_worker(topology: Topology, n_points: int, call_args: list[Any]) -> IntegrationResult:
        this_result = IntegrationResult(0., 0.)
        t_start = time.time()
        for _ in range(n_points):
            xs = [random.random() for _ in range(3)]
            weight = topology.integrand_xspace(xs, *call_args)
            if this_result.max_wgt is None or abs(weight) > abs(this_result.max_wgt):
                this_result.max_wgt = weight
                this_result.max_wgt_point = xs
            this_result.central_value += weight
            this_result.error += weight**2
            this_result.n_samples += 1
        this_result.elapsed_time += time.time() - t_start

        return this_result

    def integrate(self, t: Topology, parameterisation: str, integrand_implementation: str, improved_ltd: bool, target, **opts) -> IntegrationResult:

        integration_result = IntegrationResult(0., 0.)

        function_call_args = [
            parameterisation, integrand_implementation, improved_ltd, opts['multi_channeling']]
        for i_iter in range(opts['n_iterations']):
            logger.info(
                f'Naive integration: starting iteration {Colour.GREEN}{i_iter+1}/{opts["n_iterations"]}{Colour.END} using {Colour.BLUE}{opts["points_per_iteration"]}{Colour.END} points ...')
            if opts['n_cores'] > 1:
                n_points_per_core = opts['points_per_iteration'] // opts['n_cores']
                all_args = [(copy.deepcopy(t), n_points_per_core,
                            function_call_args), ]*(opts['n_cores']-1)
                all_args.append((copy.deepcopy(
                    t), opts['points_per_iteration'] - sum(a[1] for a in all_args), function_call_args))
                with multiprocessing.Pool(processes=opts['n_cores']) as pool:
                    all_results = pool.starmap(
                        NaiveIntegrator.naive_worker, all_args)

                # Combine results
                for result in all_results:
                    integration_result.combine_with(result)
            else:
                integration_result.combine_with(NaiveIntegrator.naive_worker(
                    copy.deepcopy(t), opts['points_per_iteration'], function_call_args))
            # Normalize a copy for temporary printout
            processed_result = copy.deepcopy(integration_result)
            processed_result.normalize()
            logger.info(
                f'... result after this iteration:\n{processed_result.str_report(target)}')

        # Normalize results
        integration_result.normalize()

        return integration_result


class VegasIntegrator(Integrator):

    def __init__(self, *args, **opts):
        super().__init__(*args, **opts)

    @staticmethod
    def vegas_worker(topology: Topology, id: int, all_xs: list[list[float]], call_args: list[Any]) -> tuple[int, list[float], IntegrationResult]:
        res = IntegrationResult(0., 0.)
        t_start = time.time()
        all_weights = []
        for xs in all_xs:
            weight = topology.integrand_xspace(xs, *call_args)
            all_weights.append(weight)
            if res.max_wgt is None or abs(weight) > abs(res.max_wgt):
                res.max_wgt = weight
                res.max_wgt_point = xs
            res.central_value += weight
            res.error += weight**2
            res.n_samples += 1
        res.elapsed_time += time.time() - t_start

        return (id, all_weights, res)

    @staticmethod
    def vegas_functor(topology: Topology, res: IntegrationResult, n_cores: int, call_args: list[Any]) -> Callable[[list[list[float]]], list[float]]:

        @vegas.batchintegrand
        def f(all_xs):
            all_weights = []
            if n_cores > 1:
                all_args = [(copy.deepcopy(topology), i_chunk, all_xs_split, call_args)
                            for i_chunk, all_xs_split in enumerate(chunks(all_xs, len(all_xs)//n_cores+1))]
                with multiprocessing.Pool(processes=n_cores) as pool:
                    all_results = pool.starmap(
                        VegasIntegrator.vegas_worker, all_args)
                for _id, wgts, this_result in sorted(all_results, key=lambda x: x[0]):
                    all_weights.extend(wgts)
                    res.combine_with(this_result)
                return all_weights
            else:
                _id, wgts, this_result = VegasIntegrator.vegas_worker(
                    topology, 0, all_xs, call_args)
                all_weights.extend(wgts)
                res.combine_with(this_result)
            return all_weights

        return f

    def integrate(self, t: Topology, parameterisation: str, integrand_implementation: str, improved_ltd: bool, _target, **opts) -> IntegrationResult:

        integration_result = IntegrationResult(0., 0.)

        integrator = vegas.Integrator(3 * [[0, 1],])

        local_worker = VegasIntegrator.vegas_functor(t, integration_result, opts['n_cores'], [
            parameterisation, integrand_implementation, improved_ltd, opts['multi_channeling']])
        # Adapt grid
        integrator(local_worker, nitn=opts['n_iterations'],
                   neval=opts['points_per_iteration'], analyzer=vegas.reporter())
        # Final result
        result = integrator(local_worker, nitn=opts['n_iterations'],
                            neval=opts['points_per_iteration'], analyzer=vegas.reporter())

        integration_result.central_value = result.mean
        integration_result.error = result.sdev
        return integration_result


class SymbolicaIntegrator(Integrator):

    def __init__(self, *args, **opts):
        super().__init__(*args, **opts)

    @staticmethod
    def symbolica_worker(topology: Topology, id: int, multi_channeling: bool, all_xs: list[SymbolicaSample], call_args: list[Any]) -> tuple[int, list[float], IntegrationResult]:
        res = IntegrationResult(0., 0.)
        t_start = time.time()
        all_weights = []
        for xs in all_xs:
            if not multi_channeling:
                weight = topology.integrand_xspace(xs.c, *(call_args+[False,]))
            else:
                weight = topology.integrand_xspace(
                    xs.c, *(call_args+[xs.d[0]]))
            all_weights.append(weight)
            if res.max_wgt is None or abs(weight) > abs(res.max_wgt):
                res.max_wgt = weight
                if not multi_channeling:
                    res.max_wgt_point = xs.c
                else:
                    res.max_wgt_point = xs.d + xs.c
            res.central_value += weight
            res.error += weight**2
            res.n_samples += 1
        res.elapsed_time += time.time() - t_start

        return (id, all_weights, res)

    @staticmethod
    def symbolica_integrand_function(topology: Topology, res: IntegrationResult, n_cores: int, multi_channeling: bool, call_args: list[Any], samples: list[Sample]) -> list[float]:
        all_weights = []
        if n_cores > 1:
            all_args = [(copy.deepcopy(topology), i_chunk, multi_channeling, [SymbolicaSample(s) for s in all_xs_split], call_args)
                        for i_chunk, all_xs_split in enumerate(chunks(samples, len(samples)//n_cores+1))]
            with multiprocessing.Pool(processes=n_cores) as pool:
                all_results = pool.starmap(
                    SymbolicaIntegrator.symbolica_worker, all_args)
            for _id, wgts, this_result in sorted(all_results, key=lambda x: x[0]):
                all_weights.extend(wgts)
                res.combine_with(this_result)
            return all_weights
        else:
            _id, wgts, this_result = SymbolicaIntegrator.symbolica_worker(
                topology, 0, multi_channeling, [SymbolicaSample(s) for s in samples], call_args)
            all_weights.extend(wgts)
            res.combine_with(this_result)
        return all_weights

    def integrate(self, t: Topology, parameterisation: str, integrand_implementation: str, improved_ltd: bool, target, **opts) -> IntegrationResult:

        integration_result = IntegrationResult(0., 0.)

        if opts['multi_channeling']:
            integrator = NumericalIntegrator.discrete([
                NumericalIntegrator.continuous(3),
                NumericalIntegrator.continuous(3),
                NumericalIntegrator.continuous(3)
            ])
        else:
            integrator = NumericalIntegrator.continuous(3)

        for i_iter in range(opts['n_iterations']):
            logger.info(
                f'Symbolica integration: starting iteration {Colour.GREEN}{i_iter+1}/{opts["n_iterations"]}{Colour.END} using {Colour.BLUE}{opts["points_per_iteration"]}{Colour.END} points ...')
            samples = integrator.sample(opts['points_per_iteration'])
            res = SymbolicaIntegrator.symbolica_integrand_function(t, integration_result, opts['n_cores'], opts['multi_channeling'], [
                parameterisation, integrand_implementation, improved_ltd], samples)
            integrator.add_training_samples(samples, res)

            # Learning rate is 1.5
            avg, err, _chi_sq = integrator.update(1.5)
            integration_result.central_value = avg
            integration_result.error = err
            logger.info(
                f'... result after this iteration:\n{integration_result.str_report(target)}')

        return integration_result
