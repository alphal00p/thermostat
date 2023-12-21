use core::panic;

use pyo3::prelude::*;

#[inline]
fn squared(v: &Vec<f64>) -> f64 {
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
}

#[inline]
fn add(v1: &Vec<f64>, v2: &Vec<f64>) -> Vec<f64> {
    return vec![v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]];
}

#[inline]
fn sub(v1: &Vec<f64>, v2: &Vec<f64>) -> Vec<f64> {
    return vec![v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]];
}

#[inline]
fn spatial(v: &Vec<f64>) -> Vec<f64> {
    return vec![v[1], v[2], v[3]];
}

#[pyfunction]
fn ltd_triangle(m_psi: f64, k: Vec<f64>, q: Vec<f64>, p: Vec<f64>) -> PyResult<f64> {
    let m_psi_sq: f64 = m_psi * m_psi;
    let ose: Vec<f64> = vec![
        (squared(&k) + m_psi_sq).sqrt(),
        ((squared(&sub(&k, &spatial(&q)))) + m_psi_sq).sqrt(),
        ((squared(&add(&k, &spatial(&p)))) + m_psi_sq).sqrt(),
    ];
    let shifts: Vec<f64> = vec![0., q[0], -p[0]];
    let mut etas: Vec<Vec<f64>> = vec![];
    for i in 0..=2 {
        let mut tmp: Vec<f64> = vec![];
        for j in 0..=2 {
            tmp.push(ose[i] + ose[j]);
        }
        etas.push(tmp);
    }
    // The algebraically equivalent LTD expression for the loop triangle diagram with spurious poles removed (imaginary part omitted)
    let res = ((2. * std::f64::consts::PI).powi(3) * (2. * ose[0]) * (2. * ose[1]) * (2. * ose[2]))
        .recip()
        * (((etas[0][2] - shifts[0] + shifts[2]) * (etas[1][2] - shifts[1] + shifts[2])).recip()
            + ((etas[0][1] + shifts[0] - shifts[1]) * (etas[1][2] - shifts[1] + shifts[2]))
                .recip()
            + ((etas[0][1] + shifts[0] - shifts[1]) * (etas[0][2] + shifts[0] - shifts[2]))
                .recip()
            + ((etas[0][2] + shifts[0] - shifts[2]) * (etas[1][2] + shifts[1] - shifts[2]))
                .recip()
            + ((etas[0][1] - shifts[0] + shifts[1]) * (etas[1][2] + shifts[1] - shifts[2]))
                .recip()
            + ((etas[0][1] - shifts[0] + shifts[1]) * (etas[0][2] - shifts[0] + shifts[2]))
                .recip());

    Ok(res)
}

#[pyfunction]
fn bubble_integrand(
    delta: f64,
    sigma: f64,
    espilon_term: f64,
    phase: f64,
    mu_r: f64,
    m_1: f64,
    m_2: f64,
    k: Vec<f64>,
    p: Vec<f64>,
) -> PyResult<f64> {
    let mut res = 0.;
    panic!("Not implemented");
    Ok(res)
}
/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn numerical_code(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ltd_triangle, m)?)?;
    m.add_function(wrap_pyfunction!(bubble_integrand, m)?)?;
    Ok(())
}
