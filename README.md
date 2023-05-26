# transpile_benchy

#### This under construction repository is a collection of various benchmark tools for testing transpilers in the field of quantum computing. It aims to provide an aggregated set of circuits worth testing transpilers against.

![Tests](https://github.com/evmckinney9/transpile_benchy/actions/workflows/tests.yml/badge.svg?branch=main)
![Format Check](https://github.com/evmckinney9/transpile_benchy/actions/workflows/format-check.yml/badge.svg?branch=main)

### Installation

I am using submodules because the dependencies are not all packaged. Therefore, after cloning the repository, you will need to run the following commands to get the submodules:

```bash
git clone --recurse-submodules
```

WIP: working on an improved modular approach to adding new submodules. All that would be required is adding a function telling the program how to navigate the submodule directory to build circuits (either from QASM or Qiskit).

TODO improve documentation

```python
run()
```

### Objective

The primary aim of this project is to simplify the process of testing new transpilation passes against a wide range of circuits. In my research studying quantum circuit optimizations, maintaining this repository deocupled from individual projects will make it much easier to evaluate and write reports. Creating a streamlined benchmark suite makes it easy to test the efficiency between multiple transpilation configurations. The repository will also feature functionalities for saving data and generating results in the form of tables and charts.

### Selected Benchmark Suite

The main benchmark suite that we will use for this project is [QASMBench](https://github.com/pnnl/QASMBench). It provides a long list of qasm files which serve as an excellent resource for running various transpilation configurations. Other submodules will be included as well, provided they contain qasm files not already included in QASMBench.

![initial_plot](images/transpile_benchy_small.svg)

### Other Existing Benchmark Suites

This project is intended for our own research purposes and is not meant to be exhaustive. However, here are some other existing benchmark suites that are available in the field:

- [MQTBench](https://github.com/cda-tum/MQTBench)
- [Arline Benchmarks](https://github.com/ArlineQ/arline_benchmarks)
- [QC-App-Oriented-Benchmarks](https://github.com/SRI-International/QC-App-Oriented-Benchmarks)
- [QUEKO-benchmark](https://github.com/tbcdebug/QUEKO-benchmark)
- [Red-Queen](https://github.com/Qiskit/red-queen/tree/main)
- [Supermarq](https://github.com/SupertechLabs/client-superstaq)

### Further Reading

- https://nonhermitian.org/posts/2021/2021-10-31-best_swap_mapper_qiskit.html
