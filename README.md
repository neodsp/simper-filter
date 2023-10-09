# simper-filter

State Variable IIR Filters for real-time audio in pure Rust.
Based on the [filter design by Andrew Simper (Cytomic)](https://cytomic.com/files/dsp/SvfLinearTrapOptimised2.pdf).

## Example

```Rust
use simper_filter::Svf;

// initialize filter with parameters
let mut filter = Svf::new(FilterType::Lowpass, 44100.0, 400.0, 0.771, 0.0);

// update filter with new parameters
filter.set(FilterType::Lowpass, 44100.0, 400.0, 0.771, 0.0);

// get response for specific frequency as complex number
let response = filter.get_response(400.);

// process a whole buffer
let input = vec![0.; 100];
let mut output = vec![0.; 100];
filter.process(&input, &mut output);

// or process on sample basis
for sample in output.iter_mut() {
    *sample = filter.tick(*sample);
}
```
