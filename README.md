# Inflammatory Aging Stem Cell Simulator

## Latest Research Output

**[View Scientific Analysis Results](https://research.lifevector.ai/inflammatory-aging-stem-cell-simulator/runs/1.0.0-20251226-073604-fe448df2/output)**

This component was automatically generated and executed by Life Vector. The link above shows the latest computational results, including data visualizations and scientific findings.

---


**Component ID:** `inflammatory-aging-stem-cell-simulator`
**Version:** 1.0.0

## What This Does

Simulates stem cell behavior under various inflammatory conditions using computational models of cellular signaling pathways. This tool allows researchers to test hypotheses and optimize experimental designs before conducting lab work.

### Key Features

- **ODE-based signaling pathway models**: Simulates JAK-STAT and NF-κB pathways affecting stem cell fate
- **Age-dependent parameter extraction**: Uses real aging transcriptome data (GSE134080) to parameterize inflammatory responses
- **Sensitivity analysis**: Identifies which parameters most strongly affect stem cell fate decisions
- **Treatment window optimization**: Predicts optimal intervention timing to preserve stem cell function
- **Uncertainty quantification**: All predictions include confidence intervals from sensitivity analysis

### Scientific Approach

This component uses:
1. **Real data from GEO (GSE134080)**: Human blood aging transcriptome with cytokine expression levels
2. **Reactome pathway data**: Cellular senescence pathway genes (R-HSA-2559583)
3. **ODE modeling**: Mathematical simulation of signaling pathway dynamics
4. **Validated parameters**: Age-related modifiers based on published aging biology

## Inputs

The component accepts optional parameters:

```python
{
    "cytokines": ["IL6", "IL8", "TNF"],  # Cytokines to simulate (default: IL6, IL8, TNF)
    "age_group": "both",                  # "young", "old", or "both" (default: both)
    "time_points": 100,                   # Simulation time points (default: 100)
    "scenarios": 5                        # Number of sensitivity scenarios (default: 5)
}
```

## Outputs

The component generates:

1. **Data Sources**: Documentation of all external data sources with accession numbers
2. **Stem Cell Quiescence Trajectories**: Time series showing fraction of quiescent stem cells
3. **Differentiation Trajectories**: Time series of stem cell differentiation rates
4. **Parameter Sensitivity Analysis**: Heatmap showing which parameters most affect outcomes
5. **Optimal Treatment Windows**: Table of predicted intervention timing for each cytokine
6. **Summary Metrics**: Key findings with uncertainty estimates
7. **Age-Dependent Cytokine Expression**: Comparison of young vs old inflammatory profiles

## Reproduce the Results

```bash
# Install dependencies
pip install -r requirements.txt

# Run the simulation
modal run modal_app.py
```

## Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=term-missing

# Run specific test class
pytest tests/test_component.py::TestHelperFunctions -v
```

## Model Details

### ODE System

The model tracks five state variables over time:
- **Q**: Quiescent stem cells (dormant, self-renewing)
- **P**: Proliferating stem cells (active cell division)
- **D**: Differentiated/senescent cells (irreversible)
- **S_JAK**: JAK-STAT pathway activation level
- **S_NFkB**: NF-κB pathway activation level

### Signaling Pathways

- **JAK-STAT**: Activated by cytokine binding, promotes proliferation
- **NF-κB**: Activated by inflammatory signals, promotes differentiation and senescence

### Age-Related Changes

The model incorporates validated age-related changes:
- Increased inflammatory signaling activation (1.6-1.8x in old)
- Reduced cell cycle entry (0.6x in old)
- Decreased quiescence maintenance (0.65 vs 0.95 in young)
- Increased differentiation propensity (0.75 vs 0.4 in young)

## Data Sources

### Primary Data
- **GSE134080**: Human blood aging transcriptome (156 samples, RNA-seq)
- **Reactome R-HSA-2559583**: Cellular senescence pathway

### Literature Parameters
Model parameters are grounded in published aging research on stem cell biology, inflammatory signaling, and cellular senescence.

## Use Cases

1. **Hypothesis testing**: Predict stem cell response to inflammatory interventions before experiments
2. **Experimental design**: Identify optimal intervention timing and dosing
3. **Parameter estimation**: Understand which biological parameters drive stem cell aging
4. **Mechanism exploration**: Simulate different signaling scenarios to understand pathway interactions

## Limitations

- This is a **computational simulation**, not experimental data
- Predictions should be **validated experimentally**
- Model simplifies complex biology to key signaling pathways
- Parameters are estimated from transcriptome data and literature, not direct measurements
- Age threshold (60 years) is arbitrary for young/old classification

## Citation

If you use this component, please cite:
- The GEO dataset: GSE134080
- Reactome Knowledgebase: https://reactome.org/
- Life Vector project: https://github.com/life-vector

## Contact

For questions or issues, please open an issue on the Life Vector GitHub repository.
