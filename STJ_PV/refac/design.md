# Subtropical Jet Finder
---

## Requirements

### Input Data
- Daily / monthly / sesasonal?
- Isobaric / isentropic
- PV exists / PV to be generated

### Output Data
- Jet position function of (hemisphere, [longitude])
    - Latitude
    - Theta

### Flexible parameters
- PV contour level
- Degree of polynomial interpolation
- Minimum latitude for fit


## Classes
### Data set class
- **Attributes**
    - Name
    - Level type
    - Data directories
    - Variable names
- **Public methods**
    - Get PV data on isentropic levels
    - Interpolate other field to PV level

### STJ Metric Class Generic
- **Attributes**
    - Name
    - Properties
    - Input data
- **Public methods**
    - Find jet
    - Output to file

### STJ IPV Gradient Metric (STJMetric)
- **Attributes**
    - Name = PVGrad
    - Input data
    - Properties
        - PV Contour
        - Poynomial fit degree
        - Min latitude for fit


- **Public methods**
    - Find jet
    - Output to file