# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Extensive changes** to the design of the package. Backwards compatibility was **not** maintained.
  - Levels no longer serve as repositories of objects. Now the model objects store collections of objects (such as nodes and component assembiles), and they feature a grid system. The grid system contains information on level elevations, as well as grids in the X-Y plane, with dedicated methods to retrieve the coordinates of intersection points. This means that now objects that previously "belonged" to a specific level now all simply belong to the model and are not associated with a particular level. Levels are now only part of the grid system, which can be used to assist element placement. This simplifies the code and allows for component assemblies that span across multiple levels (e.g. a brace spanning two stories or a column spanning two stories without an intermediate connection).
  - We now use configuration classes instead of methods with an extremely large number of arguments. We never pass functions as function arguments and dictionaries with additional arguments for those functions, which was extremely cumbersome to define. Now we instantiate an appropriate configuration object and pass that object instead. The function or method can then use the methods of that object and determine how to handle it based on its type. This allows for type aware definitions and code completion, making the user experience a lot nicer.
- Migrated to Ruff for code formatting and checking.
- Stopped using pickle to store objects to disk. Using JSON going forward, which is human-readable and safer.
- Some function/method arguments were forced to become keyword arguments to avoid the boolean trap.
- Certain terms were updated because they were inaccurate.
  - Most "generator" objects were renamed to "Creator" to avoid confusion with the meaning of "generator" in Python.

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
