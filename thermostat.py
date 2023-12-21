#!/usr/bin/env python3
from __future__ import annotations
from typing import Any, Callable, Iterator

from pprint import pprint, pformat
import argparse
import multiprocessing
import random
import logging
import time

from vectors import LorentzVector, Vector
from triangle import Triangle
from bubble import Bubble
from integrators import NaiveIntegrator, VegasIntegrator, SymbolicaIntegrator, IntegrationResult
from utils import Colour, logger, ThermostatException, Topology


def integrate(t: Topology, integrator: str, parameterisation: str, integrand_implementation: str, improved_ltd: bool, target: float | None = None, **opts) -> IntegrationResult:

    match integrator:
        case 'naive': return NaiveIntegrator().integrate(t, parameterisation, integrand_implementation, improved_ltd, target, **opts)
        case 'vegas': return VegasIntegrator().integrate(t, parameterisation, integrand_implementation, improved_ltd, target, **opts)
        case 'symbolica': return SymbolicaIntegrator().integrate(t, parameterisation, integrand_implementation, improved_ltd, target, **opts)
        case _: raise ThermostatException(f'Integrator {integrator} not implemented.')


class Plotter(object):

    def plot(self, topology: Topology, **opts):
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fixed_x = None
        for i_x in range(3):
            if i_x not in opts['xs']:
                fixed_x = i_x
                break
        if fixed_x is None:
            raise ThermostatException(
                'At least one x must be fixed (0,1 or 2).')
        n_bins = opts['mesh_size']
        # Create a grid of x and y values within the range [0., 1.]
        # Apply small offset to avoid divisions by zero
        offset = 1e-6
        x = np.linspace(opts['range'][0]+offset,
                        opts['range'][1]-offset, n_bins)
        y = np.linspace(opts['range'][0]+offset,
                        opts['range'][1]-offset, n_bins)
        X, Y = np.meshgrid(x, y)

        # Calculate the values of f(x, y) for each point in the grid
        Z = np.zeros((n_bins, n_bins))
        # Calculate the values of f(x, y) for each point in the grid using nested loops
        xs = [0.,]*3
        xs[fixed_x] = opts['fixed_x']
        for i in range(n_bins):
            for j in range(n_bins):
                xs[opts['xs'][0]] = X[i, j]
                xs[opts['xs'][1]] = Y[i, j]
                if opts['x_space']:
                    Z[i, j] = topology.integrand_xspace(
                        xs, opts['parameterisation'], opts['integrand_implementation'], opts['improved_ltd'], opts['multi_channeling'])
                else:
                    Z[i, j] = topology.integrand(Vector(
                        xs[0], xs[1], xs[2]), opts['integrand_implementation'], opts['improved_ltd'])

        # Take the logarithm of the function values, handling cases where the value is 0
        with np.errstate(divide='ignore'):
            log_Z = np.log10(np.abs(Z))
            # Replace -inf with 0 for visualization
            log_Z[log_Z == -np.inf] = 0

        if opts['x_space']:
            xs = ['x0', 'x1', 'x2']
        else:
            xs = ['kx', 'ky', 'kz']
        xs[fixed_x] = str(opts['fixed_x'])

        if not opts['3D']:
            # Create the heatmap using matplotlib
            plt.figure(figsize=(8, 6))
            plt.imshow(log_Z, origin='lower', extent=[
                       opts['range'][0], opts['range'][1], opts['range'][0], opts['range'][1]], cmap='viridis')
            plt.colorbar(label=f"log10(I({','.join(xs)}))")
        else:
            # Create a 3D plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            # Plot the surface
            surf = ax.plot_surface(X, Y, Z, cmap='viridis')
            # Add a color bar which maps values to colors
            fig.colorbar(surf, shrink=0.5, aspect=5)
            ax.set_zlabel(f"log10(I({','.join(xs)}))")

        plt.xlabel(f"{xs[opts['xs'][0]]}")
        plt.ylabel(f"{xs[opts['xs'][1]]}")
        plt.title(f"log10(I({','.join(xs)}))")
        plt.show()


if __name__ == '__main__':

    # create the top-level parser
    parser = argparse.ArgumentParser(prog='Triangler')

    # Add options common to all subcommands
    parser.add_argument('--verbosity', '-v', type=str, choices=[
                        'debug', 'info', 'critical'], default='info', help='Set verbosity level')
    parser.add_argument('--parameterisation', '-param', type=str,
                        choices=['cartesian', 'spherical'],
                        default='spherical',
                        help='Parameterisation to employ.')
    parser.add_argument('--improved_ltd', action='store_true', default=False,
                        help='Use improved LTD expression which does not suffer from numerical instabilities.')
    parser.add_argument('--integrand_implementation', '-ii', type=str, default='python', choices=[
                        'python', 'rust'], help='Integrand implementation selected. Default = %(default)s')
    parser.add_argument('--multi_channeling', '-mc', action='store_true', default=False,
                        help='Consider a multi-channeled integrand.')

    parser.add_argument('--phase', type=str,
                        choices=['real', 'imag'],
                        default='real',
                        help='Phase to compute. Default = %(default)s GeV')

    parser.add_argument('--delta', type=float,
                        default=0.5,
                        help='Delta. Default = %(default)s GeV')
    parser.add_argument('--sigma', type=float,
                        default=1.0,
                        help='Dimensionless sigma for H-function. Default = %(default)s')
    parser.add_argument('--m_1', type=float,
                        default=0.01,
                        help='First mass. Default = %(default)s GeV')
    parser.add_argument('--m_2', type=float,
                        default=0.02,
                        help='Second mass. Default = %(default)s GeV')
    parser.add_argument('--mu_r', type=float,
                        default=0.01,
                        help='Renormalization scale. Default = %(default)s GeV')
    parser.add_argument('--m_uv', type=float,
                        default=0.01,
                        help='UV regularisation mass scale. Default = %(default)s GeV')
    parser.add_argument('-p', type=float, nargs=4,
                        default=[0.005, 0.0, 0.0, 0.005],
                        help='First external. Default = %(default)s GeV')
    parser.add_argument('-q', type=float, nargs=4,
                        default=[0.005, 0.0, 0.0, -0.005],
                        help='Second external. Default = %(default)s GeV')

    parser.add_argument('--topology', '-topology', type=str,
                        choices=['triangle', 'bubble'],
                        default='bubble',
                        help='Selected topology. Default = %(default)s')

    parser.add_argument('--epsilon_expansion_term', '-eps', type=int,
                        choices=[0, -1, 2],
                        default=0,
                        help='Selected coefficient of the d-dimensional epsilon expansion to consider. Default = %(default)s')

    # Add subcommands and their options
    subparsers = parser.add_subparsers(
        title="commands", dest="command", help='Various commands available')

    # create the parser for the "inspect" command
    parser_inspect = subparsers.add_parser(
        'inspect', help='Inspect evaluation of a sample point of the integration space.')
    parser_inspect.add_argument(
        '--point', '-p', type=float, nargs=3, help='Sample point to inspect')
    parser_inspect.add_argument('--x_space', action='store_true', default=False,
                                help='Inspect a point given in x-space. Default = %(default)s')
    parser_inspect.add_argument('--full_integrand', action='store_true', default=False,
                                help='Inspect the complete integrand, incl. multi-channeling. Default = %(default)s')

    # create the parser for the "integrate" command
    parser_integrate = subparsers.add_parser(
        'integrate', help='Integrate the loop amplitude.')
    parser_integrate.add_argument('--n_iterations', '-n', type=int, default=10,
                                  help='Number of iterations to perform. Default = %(default)s')
    parser_integrate.add_argument('--points_per_iteration', '-ppi', type=int,
                                  default=100000, help='Number of points per iteration. Default = %(default)s')
    parser_integrate.add_argument('--integrator', '-it', type=str, default='vegas', choices=[
                                  'naive', 'symbolica', 'vegas'], help='Integrator selected. Default = %(default)s')
    parser_integrate.add_argument('--n_cores', '-nc', type=int, default=1,
                                  help='Number of cores to run with. Default = %(default)s')
    parser_integrate.add_argument(
        '--seed', '-s', type=int, default=None, help='Specify random seed. Default = %(default)s')

    # Create the parser for the "plot" command
    parser_plot = subparsers.add_parser('plot', help='Plot the integrand.')
    parser_plot.add_argument('--xs', type=int, nargs=2, default=None,
                             help='Chosen 2-dimension projection of the integration space')
    parser_plot.add_argument('--fixed_x', type=float, default=0.75,
                             help='Value of x kept fixed: default = %(default)s')
    parser_plot.add_argument('--range', '-r', type=float, nargs=2,
                             default=[0., 1.], help='range to plot. default = %(default)s')
    parser_plot.add_argument('--x_space', action='store_true', default=False,
                             help='Plot integrand in x-space. Default = %(default)s')
    parser_plot.add_argument('--3D', '-3D', action='store_true', default=False,
                             help='Make a 3D plot. Default = %(default)s')
    parser_plot.add_argument('--mesh_size', '-ms', type=int, default=300,
                             help='Number of bins in meshing: default = %(default)s')

    # create the parser for the "analytical_result" command
    parser_analytical_result = subparsers.add_parser(
        'analytical_result', help='Evaluate the analytical result for the amplitude.')

    args = parser.parse_args()

    match args.verbosity:
        case 'debug': logger.setLevel(logging.DEBUG)
        case 'info': logger.setLevel(logging.INFO)
        case 'critical': logger.setLevel(logging.CRITICAL)

    q_vec = LorentzVector(args.q[0], args.q[1], args.q[2], args.q[3])
    p_vec = LorentzVector(args.p[0], args.p[1], args.p[2], args.p[3])
    match args.topology:
        case 'triangle':
            topology = Triangle(args.m_2, args.m_1, q_vec, p_vec)
        case 'bubble':
            topology = Bubble(args.sigma, args.delta, args.mu_r, args.m_uv, args.m_1, args.m_2, p_vec,
                              args.epsilon_expansion_term, args.phase)
        case _:
            raise ThermostatException(
                f'Topology {args.topology} not implemented.')

    match args.command:

        case 'analytical_result':
            res = topology.analytical_result()
            logger.info(
                f'{Colour.GREEN}Analytical result:{Colour.END} {res.real:+.16e} {res.imag:+.16e}j GeV^{{-2}}')

        case 'inspect':
            if args.full_integrand:
                res = topology.integrand_xspace(
                    args.point, args.parameterisation, args.integrand_implementation, args.improved_ltd, args.multi_channeling)
                logger.info(
                    f"Full integrand evaluated at xs = [{Colour.BLUE}{', '.join(f'{xi:+.16e}' for xi in args.point)}{Colour.END}] : {Colour.GREEN}{res:+.16e}{Colour.END}")
            else:
                if args.x_space:
                    k_to_inspect, jacobian = topology.parameterize(
                        args.point, args.parameterisation)
                else:
                    k_to_inspect, jacobian = Vector(*args.point), 1.
                res = topology.integrand(
                    k_to_inspect, args.integrand_implementation, args.improved_ltd)
                report = f"Integrand evaluated at loop momentum k = [{Colour.BLUE}{', '.join(f'{ki:+.16e}' for ki in k_to_inspect.to_list())}{Colour.END}] : {Colour.GREEN}{res:+.16e}{Colour.END}"
                if args.x_space:
                    report += f' (excl. jacobian = {jacobian:+.16e})'
                logger.info(report)

        case 'integrate':
            if args.seed is not None:
                random.seed(args.seed)
                logger.info(
                    "Note that setting the random seed only ensure reproducible results with the naive integrator and a single core.")

            if args.n_cores > multiprocessing.cpu_count():
                raise ThermostatException(
                    f'Number of cores requested ({args.n_cores}) is larger than number of available cores ({multiprocessing.cpu_count()})')

            target = topology.analytical_result()
            t_start = time.time()
            res = integrate(
                topology, target=target, **vars(args)
            )
            integration_time = time.time() - t_start
            tabs = '\t'*5
            new_line = '\n'
            logger.info('-'*80)
            logger.info(f"Integration with settings below completed in {Colour.GREEN}{integration_time:.2f}s{Colour.END}:{new_line}"
                        f"{new_line.join(f'| {Colour.BLUE}{k:<30s}{Colour.END}: {Colour.GREEN}{pformat(v)}{Colour.END}' for k, v in vars(args).items())}"
                        f"{new_line}| {new_line}{res.str_report(target.real)}")
            logger.info('-'*80)

        case 'plot':
            Plotter().plot(topology, **vars(args))
        case _:
            raise ThermostatException(
                f'Command {args.command} not implemented.')
