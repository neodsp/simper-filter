# simper-svf

State Variable Filters for real-time audio in pure Rust.
Based on the [filter design by Andrew Simper (Cytomic)](https://cytomic.com/files/dsp/SvfLinearTrapOptimised2.pdf).

## Example
```Rust
use simper_svf::Svf;

let mut filter = Svf::default();
filter.set(FilterType::Lowpass, 400., 0.771, 0.);
let response = filter.get_response(400.);

let input = vec![0.; 100];
let mut output = vec![0.; 100];
filter.process(&input, &mut output);
```