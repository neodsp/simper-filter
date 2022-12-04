use num_complex::Complex64;
use num_traits::Float;
use std::f64::consts::{PI, TAU};

pub enum FilterType {
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

impl Default for FilterType {
    fn default() -> Self {
        FilterType::Lowpass
    }
}

#[derive(Default)]
struct Svf<F: Float + Default> {
    ic1eq: F,
    ic2eq: F,
    a1: F,
    a2: F,
    a3: F,
    m0: F,
    m1: F,
    m2: F,
    sample_rate: F,
    cutoff: F,
    q: F,
    gain: F,
    filter_type: FilterType,
}

impl<F: Float + Default> Svf<F> {
    pub fn process(&mut self, input: &[F], output: &mut [F]) {
        output
            .iter_mut()
            .zip(input)
            .for_each(|(out_sample, in_sample)| {
                let v0 = *in_sample;
                let v3 = v0 - self.ic2eq;
                let v1 = self.a1 * self.ic1eq + self.a2 * v3;
                let v2 = self.ic2eq + self.a2 * self.ic1eq + self.a3 * v3;

                self.ic1eq = F::from(2.).unwrap() * v1 - self.ic1eq;
                self.ic2eq = F::from(2.).unwrap() * v2 - self.ic2eq;

                *out_sample = self.m0 * v0 + self.m1 * v1 + self.m2 * v2;
            });
    }

    pub fn set(
        &mut self,
        filter_type: FilterType,
        sample_rate: u32,
        cutoff: f32,
        q: f32,
        gain: f32,
    ) {
        self.sample_rate = F::from(sample_rate).unwrap();
        self.cutoff = F::from(cutoff).unwrap();
        self.q = F::from(q).unwrap();
        self.gain = F::from(gain).unwrap();
        self.filter_type = filter_type;

        match self.filter_type {
            FilterType::Lowpass => self.lowpass(),
            FilterType::Highpass => self.highpass(),
            FilterType::Bandpass => self.bandpass(),
            FilterType::Notch => self.notch(),
            FilterType::Peak => self.peak(),
            FilterType::Allpass => self.allpass(),
            FilterType::Bell => self.bell(),
            FilterType::Lowshelf => self.lowshelf(),
            FilterType::Highshelf => self.highshelf(),
        }
    }

    pub fn get_response(&self, frequency: f64) -> Complex64 {
        match self.filter_type {
            FilterType::Lowpass => self.lowpass_response(frequency),
            FilterType::Highpass => self.highpass_response(frequency),
            FilterType::Bandpass => self.bandpass_response(frequency),
            FilterType::Notch => self.notch_response(frequency),
            FilterType::Peak => self.peak_response(frequency),
            FilterType::Allpass => self.allpass_response(frequency),
            FilterType::Bell => self.bell_response(frequency),
            FilterType::Lowshelf => self.lowshelf_response(frequency),
            FilterType::Highshelf => self.highshelf_response(frequency),
        }
    }

    pub fn lowpass(&mut self) {
        let g = F::tan(F::from(PI).unwrap() * self.cutoff / self.sample_rate);
        let k = F::one() / self.q;
        self.a1 = F::one() / (F::one() + g * (g + k));
        self.a2 = g * self.a1;
        self.a3 = g * self.a2;
        self.m0 = F::zero();
        self.m1 = F::zero();
        self.m2 = F::one();
    }

    pub fn highpass(&mut self) {
        let a = F::from(PI);
        let g = F::tan(F::from(PI).unwrap() * self.cutoff / self.sample_rate);
        let k = F::one() / self.q;
        self.a1 = F::one() / (F::one() + g * (g + k));
        self.a2 = g * self.a1;
        self.a3 = g * self.a2;
        self.m0 = F::one();
        self.m1 = -k;
        self.m2 = -F::one();
    }

    pub fn bandpass(&mut self) {
        let g = F::tan(F::from(PI).unwrap() * self.cutoff / self.sample_rate);
        let k = F::one() / self.q;
        self.a1 = F::one() / (F::one() + g * (g + k));
        self.a2 = g * self.a1;
        self.a3 = g * self.a2;
        self.m0 = F::zero();
        self.m1 = F::one();
        self.m2 = F::zero();
    }

    pub fn notch(&mut self) {
        let g = F::tan(F::from(PI).unwrap() * self.cutoff / self.sample_rate);
        let k = F::one() / self.q;
        self.a1 = F::one() / (F::one() + g * (g + k));
        self.a2 = g * self.a1;
        self.a3 = g * self.a2;
        self.m0 = F::one();
        self.m0 = F::one();
        self.m1 = -k;
        self.m2 = F::zero();
    }

    pub fn peak(&mut self) {
        let g = F::tan(F::from(PI).unwrap() * self.cutoff / self.sample_rate);
        let k = F::one() / self.q;
        self.a1 = F::one() / (F::one() + g * (g + k));
        self.a2 = g * self.a1;
        self.a3 = g * self.a2;
        self.m0 = F::one();
        self.m1 = -k;
        self.m2 = F::from(-2.).unwrap();
    }

    pub fn allpass(&mut self) {
        let g = F::tan(F::from(PI).unwrap() * self.cutoff / self.sample_rate);
        let k = F::one() / self.q;
        self.a1 = F::one() / (F::one() + g * (g + k));
        self.a2 = g * self.a1;
        self.a3 = g * self.a2;
        self.m0 = F::one();
        self.m1 = F::from(-2.).unwrap() * k;
        self.m2 = F::zero();
    }

    pub fn bell(&mut self) {
        let a = F::sqrt(self.gain);
        let g = F::tan(F::from(PI).unwrap() * self.cutoff / self.sample_rate);
        let k = F::one() / (self.q * a);
        self.a1 = F::one() / (F::one() + g * (g + k));
        self.a2 = g * self.a1;
        self.a3 = g * self.a2;
        self.m0 = F::one();
        self.m1 = k * (a * a - F::one());
        self.m2 = F::zero();
    }

    pub fn lowshelf(&mut self) {
        let a = F::sqrt(self.gain);
        let g = F::tan(F::from(PI).unwrap() * self.cutoff / self.sample_rate) / F::sqrt(a);
        let k = F::one() / self.q;
        self.a1 = F::one() / (F::one() + g * (g + k));
        self.a2 = g * self.a1;
        self.a3 = g * self.a2;
        self.m0 = F::one();
        self.m1 = k * (a - F::one());
        self.m2 = a * a - F::one();
    }

    pub fn highshelf(&mut self) {
        let a = F::sqrt(self.gain);
        let g = F::tan(F::from(PI).unwrap() * self.cutoff / self.sample_rate) * F::sqrt(a);
        let k = F::one() / self.q;
        self.a1 = F::one() / (F::one() + g * (g + k));
        self.a2 = g * self.a1;
        self.a3 = g * self.a2;
        self.m0 = a * a;
        self.m1 = k * (F::one() - a) * a;
        self.m2 = F::one() - a * a;
    }

    fn lowpass_response(&self, frequency: f64) -> Complex64 {
        let g = f64::tan(PI * self.cutoff.to_f64().unwrap() / self.sample_rate.to_f64().unwrap());
        let k = 1.0 / self.q.to_f64().unwrap();
        let f = frequency * TAU / self.sample_rate.to_f64().unwrap();
        let z = Complex64::from_polar(1.0, f);
        let response = (g * g * (1.0 + z) * (1.0 + z))
            / ((z - 1.0) * (z - 1.0) + g * g * (1.0 + z) * (1.0 + z) + g * k * (z * z - 1.0));
        response
    }

    fn highpass_response(&self, frequency: f64) -> Complex64 {
        let g = f64::tan(PI * self.cutoff.to_f64().unwrap() / self.sample_rate.to_f64().unwrap());
        let k = 1.0 / self.q.to_f64().unwrap();
        let f = frequency * TAU / self.sample_rate.to_f64().unwrap();
        let z = Complex64::from_polar(1.0, f);
        let response = ((z - 1.0) * (z - 1.0))
            / ((z - 1.0) * (z - 1.0) + g * g * (1.0 + z) * (1.0 + z) + g * k * (z * z - 1.0));
        response
    }

    fn bandpass_response(&self, frequency: f64) -> Complex64 {
        let g = f64::tan(PI * self.cutoff.to_f64().unwrap() / self.sample_rate.to_f64().unwrap());
        let k = 1.0 / self.q.to_f64().unwrap();
        let f = frequency * TAU / self.sample_rate.to_f64().unwrap();
        let z = Complex64::from_polar(1.0, f);
        let response = (g * (z * z - 1.0))
            / ((z - 1.0) * (z - 1.0) + g * g * (1.0 + z) * (1.0 + z) + g * k * (z * z - 1.0));
        response
    }

    fn notch_response(&self, frequency: f64) -> Complex64 {
        let g = f64::tan(PI * self.cutoff.to_f64().unwrap() / self.sample_rate.to_f64().unwrap());
        let k = 1.0 / self.q.to_f64().unwrap();
        let f = frequency * TAU / self.sample_rate.to_f64().unwrap();
        let z = Complex64::from_polar(1.0, f);
        let response = ((z - 1.0) * (z - 1.0) + g * g * (1.0 + z) * (1.0 + z))
            / ((z - 1.0) * (z - 1.0) + g * g * (1.0 + z) * (1.0 + z) + g * k * (z * z - 1.0));
        response
    }

    fn peak_response(&self, frequency: f64) -> Complex64 {
        let g = f64::tan(PI * self.cutoff.to_f64().unwrap() / self.sample_rate.to_f64().unwrap());
        let k = 1.0 / self.q.to_f64().unwrap();
        let f = frequency * TAU / self.sample_rate.to_f64().unwrap();
        let z = Complex64::from_polar(1.0, f);
        // Note: this is the negation of the transfer function reported in the derivation.
        let response = -((1.0 + g + (g - 1.0) * z) * (-1.0 + g + z + g * z))
            / ((z - 1.0) * (z - 1.0) + g * g * (1.0 + z) * (1.0 + z) + g * k * (z * z - 1.0));
        response
    }

    fn allpass_response(&self, frequency: f64) -> Complex64 {
        let g = f64::tan(PI * self.cutoff.to_f64().unwrap() / self.sample_rate.to_f64().unwrap());
        let k = 1.0 / self.q.to_f64().unwrap();
        let f = frequency * TAU / self.sample_rate.to_f64().unwrap();
        let z = Complex64::from_polar(1.0, f);
        let response =
            ((z - 1.0) * (z - 1.0) + g * g * (1.0 + z) * (1.0 + z) + g * (k - k * z * z))
                / ((z - 1.0) * (z - 1.0) + g * g * (1.0 + z) * (1.0 + z) + g * k * (z * z - 1.0));
        response
    }

    fn bell_response(&self, frequency: f64) -> Complex64 {
        let a = f64::sqrt(self.gain.to_f64().unwrap());
        let g = f64::tan(PI * self.cutoff.to_f64().unwrap() / self.sample_rate.to_f64().unwrap());
        let k = 1.0 / self.q.to_f64().unwrap();
        let z = Complex64::from_polar(1.0, frequency * TAU / self.sample_rate.to_f64().unwrap());
        let response = (g * k * (z * z - 1.0)
            + a * (g * (1.0 + z) * ((a * a - 1.0) * k / a * (z - 1.0))
                + ((z - 1.0) * (z - 1.0) + g * g * (1.0 + z) * (1.0 + z))))
            / (g * k * (z * z - 1.0) + a * ((z - 1.0) * (z - 1.0) + g * g * (z + 1.0) * (z + 1.0)));
        response
    }

    fn lowshelf_response(&self, frequency: f64) -> Complex64 {
        let a = f64::sqrt(self.gain.to_f64().unwrap());
        let g = f64::tan(PI * self.cutoff.to_f64().unwrap() / self.sample_rate.to_f64().unwrap());
        let k = 1.0 / self.q.to_f64().unwrap();
        let z = Complex64::from_polar(1.0, frequency * TAU / self.sample_rate.to_f64().unwrap());
        let sqrt_a = f64::sqrt(a);
        let response = (a * (z - 1.0) * (z - 1.0)
            + g * g * a * a * (z + 1.0) * (z + 1.0)
            + sqrt_a * g * a * k * (z * z - 1.0))
            / (a * (z - 1.0) * (z - 1.0)
                + g * g * (1.0 + z) * (1.0 + z)
                + sqrt_a * g * k * (z * z - 1.0));
        response
    }

    fn highshelf_response(&self, frequency: f64) -> Complex64 {
        let a = f64::sqrt(self.gain.to_f64().unwrap());
        let g = f64::tan(PI * self.cutoff.to_f64().unwrap() / self.sample_rate.to_f64().unwrap());
        let k = 1.0 / self.q.to_f64().unwrap();
        let z = Complex64::from_polar(1.0, frequency * TAU / self.sample_rate.to_f64().unwrap());
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
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let mut filter = Svf::<f32>::default();
        filter.set(FilterType::Lowpass, 44100, 400., f32::sqrt(2.0), 0.);
        filter.get_response(20.);
        let input = vec![0.; 100];
        let mut output = vec![0.; 100];
        filter.process(&input, &mut output);
    }
}
