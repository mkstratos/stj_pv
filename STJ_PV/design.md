# Subtropical Jet Finder
---

## Requirements
Requirements for STJ Finder using PV Gradient method

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
### Jet find run [JetFindRun]
- **Attributes**
    - Input data type [DSet]
    - Output data frequency [time, spatial]
    - Jet finder type
    - Jet finder input properties
    - Log
- **Public Methods**
    - Run finder
    - Output data
    - Setup Logger

--
### Data set class generic [DSet]
- **Attributes**
    - Name
    - Level type
    - Data directories
    - Variable names
- **Public methods**
    - Get PV data on isentropic levels
    - Interpolate other field(s) to PV level
    - Get zonal mean data

--
### STJ Metric Class Generic [STJMetric]
- **Attributes**
    - Name
    - Properties
    - Input data
- **Public methods**
    - Find jet
    - Output to file

--
### STJ IPV Gradient Metric (STJMetric) [STJPV]
- **Attributes**
    - Name = 'PVGrad'
    - Input data
    - Properties
        - PV Contour
        - Poynomial fit degree
        - Min latitude for fit
- **Public methods**
    - Find jet
    - Output to file