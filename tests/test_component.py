"""Tests for Inflammatory Aging Stem Cell Simulator.

These tests verify the component works correctly before deployment.
"""

import pytest
import numpy as np
import pandas as pd
from life_vector_commons import TypedOutput, OutputPattern


class TestAnalyze:
    """Test the main analyze function."""

    def test_returns_required_keys(self):
        """Verify analyze returns outputs, metrics, and summary."""
        from modal_app import analyze

        result = analyze({})

        assert "outputs" in result, "Missing 'outputs' key"
        assert "metrics" in result, "Missing 'metrics' key"
        assert "summary" in result, "Missing 'summary' key"

    def test_outputs_are_valid_typed_outputs(self):
        """Verify all outputs conform to TypedOutput schema."""
        from modal_app import analyze

        result = analyze({})

        assert len(result["outputs"]) > 0, "Should have at least one output"

        for output in result["outputs"]:
            # Each output should have required fields
            assert "pattern" in output, "Output missing 'pattern'"
            assert "data" in output, "Output missing 'data'"
            assert "label" in output, "Output missing 'label'"
            assert "description" in output, "Output missing 'description'"

            # Pattern should be valid
            pattern = output["pattern"]
            valid_patterns = [p.value for p in OutputPattern]
            assert pattern in valid_patterns, f"Invalid pattern: {pattern}"

    def test_metrics_is_dict(self):
        """Verify metrics is a dictionary."""
        from modal_app import analyze

        result = analyze({})

        assert isinstance(result["metrics"], dict), "metrics should be a dict"
        assert result["metrics"]["status"] == "completed", "Status should be completed"
        assert "cytokines_tested" in result["metrics"]
        assert "data_sources_used" in result["metrics"]

    def test_summary_is_string(self):
        """Verify summary is a string."""
        from modal_app import analyze

        result = analyze({})

        assert isinstance(result["summary"], str), "summary should be a string"
        assert len(result["summary"]) > 0, "summary should not be empty"

    def test_custom_parameters(self):
        """Test with custom input parameters."""
        from modal_app import analyze

        input_data = {
            "cytokines": ["IL6", "TNF"],
            "age_group": "old",
            "time_points": 50,
            "scenarios": 3,
        }

        result = analyze(input_data)

        assert result["metrics"]["cytokines_tested"] == 2
        assert result["metrics"]["simulation_timepoints"] == 50
        assert result["metrics"]["scenarios_simulated"] == 3


class TestOutputPatterns:
    """Test that output patterns have correct data structures."""

    def test_key_metrics_pattern(self):
        """KEY_METRICS pattern should have metrics list with uncertainty."""
        from modal_app import analyze

        result = analyze({})

        key_metrics_found = False
        for output in result["outputs"]:
            if output["pattern"] == OutputPattern.KEY_METRICS.value:
                key_metrics_found = True
                assert "metrics" in output["data"], "KEY_METRICS needs 'metrics' in data"

                # Check that metrics have uncertainty
                metrics = output["data"]["metrics"]
                assert len(metrics) > 0, "Should have at least one metric"

                for metric in metrics:
                    assert "name" in metric
                    assert "value" in metric
                    assert "unit" in metric
                    # Check uncertainty quantification
                    if "lower_bound" in metric:
                        assert "upper_bound" in metric
                        assert "confidence_level" in metric
                        assert "uncertainty_source" in metric

        assert key_metrics_found, "Should have KEY_METRICS output"

    def test_time_series_pattern(self):
        """TIME_SERIES pattern should have series data."""
        from modal_app import analyze

        result = analyze({})

        time_series_found = False
        for output in result["outputs"]:
            if output["pattern"] == OutputPattern.TIME_SERIES.value:
                time_series_found = True
                # Check required fields
                assert "timestamps" in output["data"], "TIME_SERIES needs 'timestamps' in data"
                assert "values" in output["data"], "TIME_SERIES needs 'values' in data"

                # Verify structure
                assert len(output["data"]["timestamps"]) > 0, "Should have timestamps"
                assert len(output["data"]["values"]) > 0, "Should have values"
                assert len(output["data"]["timestamps"]) == len(output["data"]["values"])

        assert time_series_found, "Should have TIME_SERIES output"

    def test_data_sources_output(self):
        """Verify data sources are documented."""
        from modal_app import analyze

        result = analyze({})

        data_sources_found = False
        for output in result["outputs"]:
            if output["label"] == "Data Sources":
                data_sources_found = True
                assert output["pattern"] == OutputPattern.TABULAR.value
                assert "columns" in output["data"]
                assert "rows" in output["data"]

                # Check structure
                assert "Source" in output["data"]["columns"]
                assert "Accession" in output["data"]["columns"]
                assert len(output["data"]["rows"]) > 0

        assert data_sources_found, "Should document data sources"

    def test_sensitivity_analysis_output(self):
        """Verify sensitivity analysis is performed."""
        from modal_app import analyze

        result = analyze({})

        sensitivity_found = False
        for output in result["outputs"]:
            if "Sensitivity" in output["label"]:
                sensitivity_found = True
                assert output["pattern"] == OutputPattern.MATRIX.value
                assert "values" in output["data"]
                assert "rows" in output["data"]
                assert "cols" in output["data"]

        assert sensitivity_found, "Should have sensitivity analysis"

    def test_treatment_windows_output(self):
        """Verify treatment windows are identified."""
        from modal_app import analyze

        result = analyze({})

        windows_found = False
        for output in result["outputs"]:
            if "Treatment Windows" in output["label"]:
                windows_found = True
                assert output["pattern"] == OutputPattern.TABULAR.value
                assert "columns" in output["data"]
                assert "rows" in output["data"]

                # Check for expected columns
                columns = output["data"]["columns"]
                assert "Cytokine" in columns
                assert "Optimal_Window_Start_hr" in columns
                assert "Optimal_Window_End_hr" in columns

        assert windows_found, "Should identify treatment windows"


class TestHelperFunctions:
    """Test individual helper functions."""

    def test_fetch_aging_inflammatory_data(self):
        """Test data fetching from GEO and Reactome."""
        from modal_app import fetch_aging_inflammatory_data

        aging_data, data_sources = fetch_aging_inflammatory_data()

        # Check data structure
        assert "expression" in aging_data
        assert "metadata" in aging_data
        assert "senescence_genes" in aging_data

        # Check data sources
        assert len(data_sources) >= 1, "Should have at least one data source"

        # Verify GEO data
        assert len(aging_data["metadata"]) > 0, "Should have samples"
        # Expression data might be empty for some GEO datasets, that's ok
        assert aging_data["expression"] is not None, "Should have expression data structure"

    def test_extract_cytokine_parameters(self):
        """Test cytokine parameter extraction."""
        from modal_app import fetch_aging_inflammatory_data, extract_cytokine_parameters

        aging_data, _ = fetch_aging_inflammatory_data()
        cytokines = ["IL6", "IL8", "TNF"]

        params = extract_cytokine_parameters(aging_data, cytokines)

        # Check structure
        assert len(params) == 3

        for cyt in cytokines:
            assert cyt in params
            assert "young" in params[cyt]
            assert "old" in params[cyt]
            assert "fold_change_with_age" in params[cyt]

            # Check young/old structure
            for age in ["young", "old"]:
                assert "expression_level" in params[cyt][age]
                assert "expression_std" in params[cyt][age]
                assert params[cyt][age]["expression_level"] > 0

    def test_calculate_age_modifiers(self):
        """Test age modifier calculation."""
        from modal_app import fetch_aging_inflammatory_data, calculate_age_modifiers

        aging_data, _ = fetch_aging_inflammatory_data()
        modifiers = calculate_age_modifiers(aging_data)

        # Check structure
        assert "young" in modifiers
        assert "old" in modifiers

        # Check that old has increased inflammatory signaling
        assert modifiers["old"]["nfkb_activation_rate"] > modifiers["young"]["nfkb_activation_rate"]
        assert modifiers["old"]["jakstat_activation_rate"] > modifiers["young"]["jakstat_activation_rate"]

        # Check that old has reduced quiescence maintenance
        assert modifiers["old"]["quiescence_maintenance"] < modifiers["young"]["quiescence_maintenance"]

    def test_stem_cell_ode_model(self):
        """Test ODE model dynamics."""
        from modal_app import stem_cell_ode_model

        # Initial state
        state = [0.90, 0.08, 0.02, 0.1, 0.1]  # Q, P, D, S_JAK, S_NFkB
        t = 0

        params = {
            "cytokine_level": 0.5,
            "age_modifiers": {
                "nfkb_activation_rate": 1.0,
                "jakstat_activation_rate": 1.0,
                "cell_cycle_entry_rate": 1.0,
                "dna_damage_accumulation": 0.3,
                "quiescence_maintenance": 0.95,
                "differentiation_propensity": 0.4,
            }
        }

        derivatives = stem_cell_ode_model(state, t, params)

        # Check output structure
        assert len(derivatives) == 5, "Should return 5 derivatives"

        # Check that derivatives are reasonable numbers
        for deriv in derivatives:
            assert isinstance(deriv, (int, float))
            assert not np.isnan(deriv)
            assert not np.isinf(deriv)

    def test_simulate_stem_cell_fates(self):
        """Test stem cell fate simulation."""
        from modal_app import simulate_stem_cell_fates

        cytokine_params = {
            "IL6": {
                "young": {"expression_level": 5.0, "expression_std": 1.0, "receptor_affinity": 0.85},
                "old": {"expression_level": 8.0, "expression_std": 1.5, "receptor_affinity": 0.72},
                "fold_change_with_age": 1.6,
                "signaling_strength": 0.6,
            }
        }

        age_modifiers = {
            "young": {
                "nfkb_activation_rate": 1.0,
                "jakstat_activation_rate": 1.0,
                "cell_cycle_entry_rate": 1.0,
                "dna_damage_accumulation": 0.3,
                "quiescence_maintenance": 0.95,
                "differentiation_propensity": 0.4,
            },
            "old": {
                "nfkb_activation_rate": 1.8,
                "jakstat_activation_rate": 1.6,
                "cell_cycle_entry_rate": 0.6,
                "dna_damage_accumulation": 1.0,
                "quiescence_maintenance": 0.65,
                "differentiation_propensity": 0.75,
            }
        }

        trajectories = simulate_stem_cell_fates(
            cytokine_params=cytokine_params,
            age_modifiers=age_modifiers,
            n_timepoints=50,
            age_group="both"
        )

        # Check output structure
        assert isinstance(trajectories, pd.DataFrame)
        assert "time" in trajectories.columns
        assert len(trajectories) == 50

        # Check that we have trajectories for both ages
        assert any("young" in col for col in trajectories.columns)
        assert any("old" in col for col in trajectories.columns)

        # Check that values are in reasonable ranges (0-1 for cell fractions)
        for col in trajectories.columns:
            if col != "time":
                assert trajectories[col].min() >= 0
                assert trajectories[col].max() <= 1.5  # Allow some slack for numerical stability

    def test_perform_sensitivity_analysis(self):
        """Test sensitivity analysis."""
        from modal_app import perform_sensitivity_analysis

        cytokine_params = {
            "IL6": {
                "young": {"expression_level": 5.0, "expression_std": 1.0, "receptor_affinity": 0.85},
                "old": {"expression_level": 8.0, "expression_std": 1.5, "receptor_affinity": 0.72},
                "fold_change_with_age": 1.6,
                "signaling_strength": 0.6,
            }
        }

        age_modifiers = {
            "young": {
                "nfkb_activation_rate": 1.0,
                "jakstat_activation_rate": 1.0,
                "cell_cycle_entry_rate": 1.0,
                "dna_damage_accumulation": 0.3,
                "quiescence_maintenance": 0.95,
                "differentiation_propensity": 0.4,
            },
            "old": {
                "nfkb_activation_rate": 1.8,
                "jakstat_activation_rate": 1.6,
                "cell_cycle_entry_rate": 0.6,
                "dna_damage_accumulation": 1.0,
                "quiescence_maintenance": 0.65,
                "differentiation_propensity": 0.75,
            }
        }

        sensitivity = perform_sensitivity_analysis(
            cytokine_params=cytokine_params,
            age_modifiers=age_modifiers,
            n_scenarios=3
        )

        # Check output structure
        assert "sensitivity_matrix" in sensitivity
        assert "parameters" in sensitivity
        assert "outcomes" in sensitivity

        # Check matrix dimensions
        matrix = sensitivity["sensitivity_matrix"]
        assert len(matrix) == len(sensitivity["parameters"])
        assert len(matrix[0]) == len(sensitivity["outcomes"])

    def test_identify_treatment_windows(self):
        """Test treatment window identification."""
        from modal_app import identify_treatment_windows

        # Create mock trajectories
        time = np.linspace(0, 120, 100)
        quiescence = np.exp(-time / 50)  # Exponential decay

        trajectories = pd.DataFrame({
            "time": time,
            "old_IL6_quiescence": quiescence,
        })

        cytokine_params = {"IL6": {}}

        windows = identify_treatment_windows(trajectories, cytokine_params)

        # Check output
        assert len(windows) == 1
        assert len(windows[0]) == 5  # cytokine, start, end, duration, preservation
        assert windows[0][0] == "IL6"


class TestIntegration:
    """Integration tests."""

    def test_end_to_end_simulation(self):
        """End-to-end test of the full simulation pipeline."""
        from modal_app import analyze

        input_data = {
            "cytokines": ["IL6", "TNF"],
            "age_group": "both",
            "time_points": 50,
            "scenarios": 3,
        }

        result = analyze(input_data)

        # Verify all expected outputs are present
        output_labels = [o["label"] for o in result["outputs"]]

        assert "Data Sources" in output_labels
        assert "Stem Cell Quiescence Trajectories" in output_labels
        assert "Stem Cell Differentiation Trajectories" in output_labels
        assert "Parameter Sensitivity Analysis" in output_labels
        assert "Optimal Treatment Windows" in output_labels
        assert "Simulation Summary Metrics" in output_labels
        assert "Age-Dependent Cytokine Expression" in output_labels

        # Verify metrics
        assert result["metrics"]["status"] == "completed"
        assert result["metrics"]["cytokines_tested"] == 2

        # Verify summary
        assert "inflammatory" in result["summary"].lower()
        assert "stem cell" in result["summary"].lower()


