# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Some function/method arguments were forced to become keyword arguments only. If you get an error related to this, inspect the applicable method and change the syntax to use keyword arguments.
- Migrated to Ruff for code formatting and checking.

### Removed

- Removed the `print` method from analysis objects, since it was largely unused.

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
