#![cfg_attr(not(test), no_std)]

use core::f64::consts::{PI, TAU};
use num_complex::Complex64;
use num_traits::Float;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SvfError {
    #[error("the sample rate must be set first")]
    NoSampleRate,
    #[error("the frequency is higher than nyqist")]
    FrequencyOverNyqist,
    #[error("the frequency is 0 or lower than 0")]
    FrequencyTooLow,
    #[error("q is lower than zero")]
    NegativeQ,
    #[error("fatal number conversion error")]
    Fatal,
}

/// All available filter types for the State Variable Filter
#[derive(Default, Clone, PartialEq, Eq)]
pub enum FilterType {
    #[default]
    Lowpass,
    Highpass,
    Bandpass,
    Notch,
    Peak,
    Allpass,
    Bell,
    Lowshelf,
    Highshelf,
}

/// A State Variable Filter that is copied from Andrew Simper (Cytomic)
/// https://cytomic.com/files/dsp/SvfLinearTrapOptimised2.pdf
#[derive(Default)]
pub struct Svf<F: Float> {
    coefficients: SvfCoefficients<F>,
    ic1eq: F,
    ic2eq: F,
}

/// The filter coefficients that hold old necessary data to reproduce the filter
#[derive(Default, Clone, PartialEq, Eq)]
pub struct SvfCoefficients<F: Float> {
    filter_type: FilterType,
    sample_rate: F,
    cutoff: F,
    gain: F,
    q: F,
    a1: F,
    a2: F,
    a3: F,
    m0: F,
    m1: F,
    m2: F,
}

impl<F: Float + Default> Svf<F> {
    /// sets the filter to an inital response
    /// use `Svf::default` otherwise
    /// Parameters:
    /// - filter_type: choose one of the filter types, like peak, lowpass or highpass
    /// - sample_rate: the sample_rate of the audio buffer that the filter should be applied on
    /// - frequency: the frequency in Hz where the cutoff of the filter should be
    /// - q: the steepness of the filter
    /// - gain: the gain boost or decrease of the filter
    pub fn new(
        filter_type: FilterType,
        sample_rate: F,
        cutoff: F,
        q: F,
        gain: F,
    ) -> Result<Self, SvfError> {
        let mut svf = Self::default();
        svf.set(filter_type, sample_rate, cutoff, q, gain)?;
        Ok(svf)
    }

    /// process helper function that calls `Svf::tick` on each sample
    #[inline]
    pub fn process(&mut self, input: &[F], output: &mut [F]) {
        output
            .iter_mut()
            .zip(input)
            .for_each(|(out_sample, in_sample)| {
                *out_sample = self.tick(*in_sample);
            });
    }

    /// Reset state of filter.
    /// Can be used when the audio callback is restarted.
    #[inline]
    pub fn reset(&mut self) {
        self.ic1eq = F::zero();
        self.ic2eq = F::zero();
    }

    /// The set function is setting the current parameters of the filter.
    /// Don't call from a different thread than tick.
    /// Parameters:
    /// - filter_type: choose one of the filter types, like peak, lowpass or highpass
    /// - sample_rate: the sample_rate of the audio buffer that the filter should be applied on
    /// - frequency: the frequency in Hz where the cutoff of the filter should be
    /// - q: the steepness of the filter
    /// - gain: the gain boost or decrease of the filter
    #[inline]
    pub fn set(
        &mut self,
        filter_type: FilterType,
        sample_rate: F,
        cutoff: F,
        q: F,
        gain: F,
    ) -> Result<(), SvfError> {
        self.coefficients
            .set(filter_type, sample_rate, cutoff, q, gain)
    }

    /// Set new filter parameters from coefficients struct
    pub fn set_coeffs(&mut self, coefficients: SvfCoefficients<F>) {
        self.coefficients = coefficients;
    }

    /// The process that is applying the filter on a sample by sample basis
    /// `process` is a utility function to call tick on a full frame
    #[inline]
    pub fn tick(&mut self, input: F) -> F {
        let v0 = input;
        let v3 = v0 - self.ic2eq;
        let v1 = self.coefficients.a1 * self.ic1eq + self.coefficients.a2 * v3;
        let v2 = self.ic2eq + self.coefficients.a2 * self.ic1eq + self.coefficients.a3 * v3;
        let two = F::one() + F::one();
        self.ic1eq = two * v1 - self.ic1eq;
        self.ic2eq = two * v2 - self.ic2eq;

        self.coefficients.m0 * v0 + self.coefficients.m1 * v1 + self.coefficients.m2 * v2
    }

    /// get a reference to the coefficients
    /// can be used to clone it to the UI thread, to plot the response
    pub fn coefficients(&self) -> &SvfCoefficients<F> {
        &self.coefficients
    }

    /// utility method to get the response
    /// should not be called from another thread than the set methods
    pub fn get_response(&self, frequency: f64) -> Result<Complex64, SvfError> {
        SvfResponse::response(&self.coefficients, frequency)
    }
}

impl<F: Float> SvfCoefficients<F> {
    /// The set function is setting the current parameters of the filter.
    /// Don't call from a different thread than tick.
    /// Parameters:
    /// - filter_type: choose one of the filter types, like peak, lowpass or highpass
    /// - sample_rate: the sample_rate of the audio buffer that the filter should be applied on
    /// - frequency: the frequency in Hz where the cutoff of the filter should be
    /// - q: the steepness of the filter
    /// - gain: the gain boost or decrease of the filter
    pub fn set(
        &mut self,
        filter_type: FilterType,
        sample_rate: F,
        cutoff_frequency: F,
        q: F,
        gain: F,
    ) -> Result<(), SvfError> {
        if q < F::zero() {
            return Err(SvfError::NegativeQ);
        }
        if cutoff_frequency > sample_rate / F::from(2.0).unwrap() {
            return Err(SvfError::FrequencyOverNyqist);
        }
        self.filter_type = filter_type;
        self.sample_rate = sample_rate;
        self.cutoff = cutoff_frequency;
        self.q = q;
        self.gain = gain;

        match self.filter_type {
            FilterType::Lowpass => {
                let g =
                    F::tan(F::from(PI).ok_or(SvfError::Fatal)? * self.cutoff / self.sample_rate);
                let k = F::one() / self.q;
                self.a1 = F::one() / (F::one() + g * (g + k));
                self.a2 = g * self.a1;
                self.a3 = g * self.a2;
                self.m0 = F::zero();
                self.m1 = F::zero();
                self.m2 = F::one();
            }
            FilterType::Highpass => {
                let g =
                    F::tan(F::from(PI).ok_or(SvfError::Fatal)? * self.cutoff / self.sample_rate);
                let k = F::one() / self.q;
                self.a1 = F::one() / (F::one() + g * (g + k));
                self.a2 = g * self.a1;
                self.a3 = g * self.a2;
                self.m0 = F::one();
                self.m1 = -k;
                self.m2 = -F::one();
            }
            FilterType::Bandpass => {
                let g =
                    F::tan(F::from(PI).ok_or(SvfError::Fatal)? * self.cutoff / self.sample_rate);
                let k = F::one() / self.q;
                self.a1 = F::one() / (F::one() + g * (g + k));
                self.a2 = g * self.a1;
                self.a3 = g * self.a2;
                self.m0 = F::zero();
                self.m1 = F::one();
                self.m2 = F::zero();
            }
            FilterType::Notch => {
                let g =
                    F::tan(F::from(PI).ok_or(SvfError::Fatal)? * self.cutoff / self.sample_rate);
                let k = F::one() / self.q;
                self.a1 = F::one() / (F::one() + g * (g + k));
                self.a2 = g * self.a1;
                self.a3 = g * self.a2;
                self.m0 = F::one();
                self.m0 = F::one();
                self.m1 = -k;
                self.m2 = F::zero();
            }
            FilterType::Peak => {
                let g =
                    F::tan(F::from(PI).ok_or(SvfError::Fatal)? * self.cutoff / self.sample_rate);
                let k = F::one() / self.q;
                self.a1 = F::one() / (F::one() + g * (g + k));
                self.a2 = g * self.a1;
                self.a3 = g * self.a2;
                self.m0 = F::one();
                self.m1 = -k;
                self.m2 = F::from(-2.).ok_or(SvfError::Fatal)?;
            }
            FilterType::Allpass => {
                let g =
                    F::tan(F::from(PI).ok_or(SvfError::Fatal)? * self.cutoff / self.sample_rate);
                let k = F::one() / self.q;
                self.a1 = F::one() / (F::one() + g * (g + k));
                self.a2 = g * self.a1;
                self.a3 = g * self.a2;
                self.m0 = F::one();
                self.m1 = F::from(-2.).ok_or(SvfError::Fatal)? * k;
                self.m2 = F::zero();
            }
            FilterType::Bell => {
                let a = F::powf(
                    F::from(10).ok_or(SvfError::Fatal)?,
                    self.gain / F::from(40).unwrap(),
                );
                let g =
                    F::tan(F::from(PI).ok_or(SvfError::Fatal)? * self.cutoff / self.sample_rate);
                let k = F::one() / (self.q * a);
                self.a1 = F::one() / (F::one() + g * (g + k));
                self.a2 = g * self.a1;
                self.a3 = g * self.a2;
                self.m0 = F::one();
                self.m1 = k * (a * a - F::one());
                self.m2 = F::zero();
            }
            FilterType::Lowshelf => {
                let a = F::powf(
                    F::from(10).ok_or(SvfError::Fatal)?,
                    self.gain / F::from(40).unwrap(),
                );
                let g =
                    F::tan(F::from(PI).ok_or(SvfError::Fatal)? * self.cutoff / self.sample_rate)
                        / F::sqrt(a);
                let k = F::one() / self.q;
                self.a1 = F::one() / (F::one() + g * (g + k));
                self.a2 = g * self.a1;
                self.a3 = g * self.a2;
                self.m0 = F::one();
                self.m1 = k * (a - F::one());
                self.m2 = a * a - F::one();
            }
            FilterType::Highshelf => {
                let a = F::powf(
                    F::from(10).ok_or(SvfError::Fatal)?,
                    self.gain / F::from(40).unwrap(),
                );
                let g =
                    F::tan(F::from(PI).ok_or(SvfError::Fatal)? * self.cutoff / self.sample_rate)
                        * F::sqrt(a);
                let k = F::one() / self.q;
                self.a1 = F::one() / (F::one() + g * (g + k));
                self.a2 = g * self.a1;
                self.a3 = g * self.a2;
                self.m0 = a * a;
                self.m1 = k * (F::one() - a) * a;
                self.m2 = F::one() - a * a;
            }
        }
        Ok(())
    }
}

/// Coefficients can be copied in the UI Thread, so that the response can be shown
pub struct SvfResponse;
impl SvfResponse {
    /// Retrieves the filter response of the current settings for a specific frequency
    /// Can be used to plot the filter magnitue and phase
    pub fn response<F: Float>(
        coeffs: &SvfCoefficients<F>,
        frequency: f64,
    ) -> Result<Complex64, SvfError> {
        if frequency <= 0.0 {
            return Err(SvfError::FrequencyTooLow);
        }
        let nyquist = coeffs.sample_rate.to_f64().ok_or(SvfError::Fatal)? / 2.0;
        if frequency > nyquist {
            return Err(SvfError::FrequencyOverNyqist);
        }

        match coeffs.filter_type {
            FilterType::Lowpass => SvfResponse::lowpass_response(coeffs, frequency),
            FilterType::Highpass => SvfResponse::highpass_response(coeffs, frequency),
            FilterType::Bandpass => SvfResponse::bandpass_response(coeffs, frequency),
            FilterType::Notch => SvfResponse::notch_response(coeffs, frequency),
            FilterType::Peak => SvfResponse::peak_response(coeffs, frequency),
            FilterType::Allpass => SvfResponse::allpass_response(coeffs, frequency),
            FilterType::Bell => SvfResponse::bell_response(coeffs, frequency),
            FilterType::Lowshelf => SvfResponse::lowshelf_response(coeffs, frequency),
            FilterType::Highshelf => SvfResponse::highshelf_response(coeffs, frequency),
        }
    }

    fn lowpass_response<F: Float>(
        coeffs: &SvfCoefficients<F>,
        frequency: f64,
    ) -> Result<Complex64, SvfError> {
        let g = f64::tan(
            PI * coeffs.cutoff.to_f64().ok_or(SvfError::Fatal)?
                / coeffs.sample_rate.to_f64().ok_or(SvfError::Fatal)?,
        );
        let k = 1.0 / coeffs.q.to_f64().ok_or(SvfError::Fatal)?;
        let f = frequency * TAU / coeffs.sample_rate.to_f64().ok_or(SvfError::Fatal)?;
        let z = Complex64::from_polar(1.0, f);
        let response = (g * g * (1.0 + z) * (1.0 + z))
            / ((z - 1.0) * (z - 1.0) + g * g * (1.0 + z) * (1.0 + z) + g * k * (z * z - 1.0));
        Ok(response)
    }

    fn highpass_response<F: Float>(
        coeffs: &SvfCoefficients<F>,
        frequency: f64,
    ) -> Result<Complex64, SvfError> {
        let g = f64::tan(
            PI * coeffs.cutoff.to_f64().ok_or(SvfError::Fatal)?
                / coeffs.sample_rate.to_f64().ok_or(SvfError::Fatal)?,
        );
        let k = 1.0 / coeffs.q.to_f64().ok_or(SvfError::Fatal)?;
        let f = frequency * TAU / coeffs.sample_rate.to_f64().ok_or(SvfError::Fatal)?;
        let z = Complex64::from_polar(1.0, f);
        let response = ((z - 1.0) * (z - 1.0))
            / ((z - 1.0) * (z - 1.0) + g * g * (1.0 + z) * (1.0 + z) + g * k * (z * z - 1.0));
        Ok(response)
    }

    fn bandpass_response<F: Float>(
        coeffs: &SvfCoefficients<F>,
        frequency: f64,
    ) -> Result<Complex64, SvfError> {
        let g = f64::tan(
            PI * coeffs.cutoff.to_f64().ok_or(SvfError::Fatal)?
                / coeffs.sample_rate.to_f64().ok_or(SvfError::Fatal)?,
        );
        let k = 1.0 / coeffs.q.to_f64().ok_or(SvfError::Fatal)?;
        let f = frequency * TAU / coeffs.sample_rate.to_f64().ok_or(SvfError::Fatal)?;
        let z = Complex64::from_polar(1.0, f);
        let response = (g * (z * z - 1.0))
            / ((z - 1.0) * (z - 1.0) + g * g * (1.0 + z) * (1.0 + z) + g * k * (z * z - 1.0));
        Ok(response)
    }

    fn notch_response<F: Float>(
        coeffs: &SvfCoefficients<F>,
        frequency: f64,
    ) -> Result<Complex64, SvfError> {
        let g = f64::tan(
            PI * coeffs.cutoff.to_f64().ok_or(SvfError::Fatal)?
                / coeffs.sample_rate.to_f64().ok_or(SvfError::Fatal)?,
        );
        let k = 1.0 / coeffs.q.to_f64().ok_or(SvfError::Fatal)?;
        let f = frequency * TAU / coeffs.sample_rate.to_f64().ok_or(SvfError::Fatal)?;
        let z = Complex64::from_polar(1.0, f);
        let response = ((z - 1.0) * (z - 1.0) + g * g * (1.0 + z) * (1.0 + z))
            / ((z - 1.0) * (z - 1.0) + g * g * (1.0 + z) * (1.0 + z) + g * k * (z * z - 1.0));
        Ok(response)
    }

    fn peak_response<F: Float>(
        coeffs: &SvfCoefficients<F>,
        frequency: f64,
    ) -> Result<Complex64, SvfError> {
        let g = f64::tan(
            PI * coeffs.cutoff.to_f64().ok_or(SvfError::Fatal)?
                / coeffs.sample_rate.to_f64().ok_or(SvfError::Fatal)?,
        );
        let k = 1.0 / coeffs.q.to_f64().ok_or(SvfError::Fatal)?;
        let f = frequency * TAU / coeffs.sample_rate.to_f64().ok_or(SvfError::Fatal)?;
        let z = Complex64::from_polar(1.0, f);
        // Note: this is the negation of the transfer function reported in the derivation.
        let response = -((1.0 + g + (g - 1.0) * z) * (-1.0 + g + z + g * z))
            / ((z - 1.0) * (z - 1.0) + g * g * (1.0 + z) * (1.0 + z) + g * k * (z * z - 1.0));
        Ok(response)
    }

    fn allpass_response<F: Float>(
        coeffs: &SvfCoefficients<F>,
        frequency: f64,
    ) -> Result<Complex64, SvfError> {
        let g = f64::tan(
            PI * coeffs.cutoff.to_f64().ok_or(SvfError::Fatal)?
                / coeffs.sample_rate.to_f64().ok_or(SvfError::Fatal)?,
        );
        let k = 1.0 / coeffs.q.to_f64().ok_or(SvfError::Fatal)?;
        let f = frequency * TAU / coeffs.sample_rate.to_f64().ok_or(SvfError::Fatal)?;
        let z = Complex64::from_polar(1.0, f);
        let response =
            ((z - 1.0) * (z - 1.0) + g * g * (1.0 + z) * (1.0 + z) + g * (k - k * z * z))
                / ((z - 1.0) * (z - 1.0) + g * g * (1.0 + z) * (1.0 + z) + g * k * (z * z - 1.0));
        Ok(response)
    }

    fn bell_response<F: Float>(
        coeffs: &SvfCoefficients<F>,
        frequency: f64,
    ) -> Result<Complex64, SvfError> {
        let a = f64::sqrt(coeffs.gain.to_f64().ok_or(SvfError::Fatal)?);
        let g = f64::tan(
            PI * coeffs.cutoff.to_f64().ok_or(SvfError::Fatal)?
                / coeffs.sample_rate.to_f64().ok_or(SvfError::Fatal)?,
        );
        let k = 1.0 / coeffs.q.to_f64().ok_or(SvfError::Fatal)?;
        let z = Complex64::from_polar(
            1.0,
            frequency * TAU / coeffs.sample_rate.to_f64().ok_or(SvfError::Fatal)?,
        );
        let response = (g * k * (z * z - 1.0)
            + a * (g * (1.0 + z) * ((a * a - 1.0) * k / a * (z - 1.0))
                + ((z - 1.0) * (z - 1.0) + g * g * (1.0 + z) * (1.0 + z))))
            / (g * k * (z * z - 1.0) + a * ((z - 1.0) * (z - 1.0) + g * g * (z + 1.0) * (z + 1.0)));
        Ok(response)
    }

    fn lowshelf_response<F: Float>(
        coeffs: &SvfCoefficients<F>,
        frequency: f64,
    ) -> Result<Complex64, SvfError> {
        let a = f64::sqrt(coeffs.gain.to_f64().ok_or(SvfError::Fatal)?);
        let g = f64::tan(
            PI * coeffs.cutoff.to_f64().ok_or(SvfError::Fatal)?
                / coeffs.sample_rate.to_f64().ok_or(SvfError::Fatal)?,
        );
        let k = 1.0 / coeffs.q.to_f64().ok_or(SvfError::Fatal)?;
        let z = Complex64::from_polar(
            1.0,
            frequency * TAU / coeffs.sample_rate.to_f64().ok_or(SvfError::Fatal)?,
        );
        let sqrt_a = f64::sqrt(a);
        let response = (a * (z - 1.0) * (z - 1.0)
            + g * g * a * a * (z + 1.0) * (z + 1.0)
            + sqrt_a * g * a * k * (z * z - 1.0))
            / (a * (z - 1.0) * (z - 1.0)
                + g * g * (1.0 + z) * (1.0 + z)
                + sqrt_a * g * k * (z * z - 1.0));
        Ok(response)
    }

    fn highshelf_response<F: Float>(
        coeffs: &SvfCoefficients<F>,
        frequency: f64,
    ) -> Result<Complex64, SvfError> {
        let a = f64::sqrt(coeffs.gain.to_f64().ok_or(SvfError::Fatal)?);
        let g = f64::tan(
            PI * coeffs.cutoff.to_f64().ok_or(SvfError::Fatal)?
                / coeffs.sample_rate.to_f64().ok_or(SvfError::Fatal)?,
        );
        let k = 1.0 / coeffs.q.to_f64().ok_or(SvfError::Fatal)?;
        let z = Complex64::from_polar(
            1.0,
            frequency * TAU / coeffs.sample_rate.to_f64().ok_or(SvfError::Fatal)?,
        );
        let sqrt_a = f64::sqrt(a);
        let response = (sqrt_a
            * g
            * (1.0 + z)
            * (-(a - 1.0) * a * k * (z - 1.0) + sqrt_a * g * (1.0 - a * a) * (1.0 + z))
            + a * a
                * ((z - 1.0) * (z - 1.0)
                    + a * g * g * (1.0 + z) * (1.0 + z)
                    + sqrt_a * g * k * (z * z - 1.0)))
            / ((z - 1.0) * (z - 1.0)
                + a * g * g * (1.0 + z) * (1.0 + z)
                + sqrt_a * g * k * (z * z - 1.0));
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let mut filter = Svf::<f32>::default();
        filter
            .set(FilterType::Lowpass, 44100.0, 400.0, f32::sqrt(2.0), 0.0)
            .unwrap();
        let _response = filter.get_response(20.0).unwrap();
        let input = vec![0.0; 100];
        let mut output = vec![0.0; 100];
        filter.process(&input, &mut output);
    }
}
