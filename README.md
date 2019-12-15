# Autovergleich

Scrapes a list of electric cars from www.autoscout24.ch and models price by age and milage.

_Note: as of August 2019 Autoscout24.ch blocks scraping attempts._

Example:


![Scatter plot](plot/scatter_days_Preis.png?raw=true "Scatter plot")
![Regression plot](plot/regplot_days_Preis.png?raw=true "Regression plot")

```
                  Mixed Linear Model Regression Results
=========================================================================
Model:                  MixedLM     Dependent Variable:     Preis        
No. Observations:       2259        Method:                 REML         
No. Groups:             20          Scale:                  24895560.6918
Min. group size:        1           Likelihood:             -22485.1992  
Max. group size:        669         Converged:              Yes          
Mean group size:        113.0                                            
-------------------------------------------------------------------------
                     Coef.     Std.Err.    z    P>|z|   [0.025    0.975] 
-------------------------------------------------------------------------
Intercept            34250.926 2392.539  14.316 0.000 29561.636 38940.215
Kilometer               -0.049    0.010  -4.782 0.000    -0.069    -0.029
Inverkehrsetzung        -8.476    0.260 -32.573 0.000    -8.986    -7.966
Group Var        109873272.345 7426.142                                  
=========================================================================
```

### Prerequisites
```
statsmodels
requests
matplotlib
pandas
numpy
seaborn
beautifulsoup4
scikit_learn
```
