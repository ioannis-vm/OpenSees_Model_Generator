# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]


## [0.2.7] - 2024-08-25

### Fixed

- Fixed bug in applying gravity loads before a time-history or pushover analysis.
- Replaced pushover convergence tolerance with a more appropriate value.

## [0.2.6] - 2024-05-01

### Fixed

Fixed node restraints and time history analysis acceleration inputs for `opensees`.

## [0.2.5] - 2024-05-01

### Added

The convergence test tolerance can now be passed as an optional argument for time-history analyses (`test_tolerance`). It defaults to `1e-12`.
