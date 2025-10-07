import pytest
import tempfile
import json
import os
from symxplorer.spice_engine.storage import (
    Spice_Simulation_Point,
    Spice_Simulation_Database,
)
    
# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def sample_point():
    return Spice_Simulation_Point(
        parameters={"W": 1e-6, "L": 0.18e-6},
        scalarized_metric=0.92,
        metrics={"gain": 60.1, "bandwidth": 2.3e6},
    )

@pytest.fixture
def db_with_schema():
    db = Spice_Simulation_Database(parameter_names=["W", "L"], metric_names=["gain", "bandwidth"])
    return db


# -----------------------------
# Core Functionality Tests
# -----------------------------
def test_schema_inference(sample_point):
    db = Spice_Simulation_Database()
    db.add_point(sample_point)
    assert db.parameter_names == ["W", "L"]
    assert db.metric_names == ["gain", "bandwidth"]
    assert len(db.points) == 1


def test_schema_enforcement_pass(db_with_schema, sample_point):
    db_with_schema.add_point(sample_point)
    assert len(db_with_schema.points) == 1


def test_schema_enforcement_fail_on_parameters(db_with_schema):
    bad_point = Spice_Simulation_Point(
        parameters={"W": 1e-6},  # Missing "L"
        scalarized_metric=0.5,
        metrics={"gain": 55.0, "bandwidth": 1.8e6},
    )
    with pytest.raises(ValueError, match="Parameter mismatch"):
        db_with_schema.add_point(bad_point)


def test_schema_enforcement_fail_on_metrics(db_with_schema):
    bad_point = Spice_Simulation_Point(
        parameters={"W": 1e-6, "L": 0.18e-6},
        scalarized_metric=0.5,
        metrics={"gain": 55.0},  # Missing bandwidth
    )
    with pytest.raises(ValueError, match="Metric mismatch"):
        db_with_schema.add_point(bad_point)


def test_best_point_returns_highest_scalar(sample_point):
    db = Spice_Simulation_Database()
    db.add_point(sample_point)
    p2 = Spice_Simulation_Point(
        parameters={"W": 2e-6, "L": 0.18e-6},
        scalarized_metric=0.5,
        metrics={"gain": 62.0, "bandwidth": 2.1e6},
    )
    db.add_point(p2)
    best = db.best_point()
    assert best.scalarized_metric == 0.92


# -----------------------------
# I/O Serialization Tests
# -----------------------------
def test_to_and_from_json(sample_point):
    db = Spice_Simulation_Database()
    db.add_point(sample_point)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        db.to_json(tmp.name)
        tmp_path = tmp.name

    db2 = Spice_Simulation_Database.load_json(tmp_path)
    assert db2.parameter_names == db.parameter_names
    assert len(db2.points) == len(db.points)
    assert db2.points[0].parameters == db.points[0].parameters

    os.remove(tmp_path)


def test_load_json_invalid_format(tmp_path=None):
    """Test that invalid JSON raises an error."""
    bad_data = {"not_points": []}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as tmp:
        json.dump(bad_data, tmp)
        tmp_path = tmp.name

    with pytest.raises(ValueError, match="Invalid database format"):
        Spice_Simulation_Database.load_json(tmp_path)

    os.remove(tmp_path)


def test_to_numpy_shape(sample_point):
    db = Spice_Simulation_Database()
    db.add_point(sample_point)
    p2 = Spice_Simulation_Point(
        parameters={"W": 2e-6, "L": 0.18e-6},
        scalarized_metric=0.85,
        metrics={"gain": 62.0, "bandwidth": 2.1e6},
    )
    db.add_point(p2)

    params, metrics, scalars = db.to_numpy()
    assert params.shape == (2, 2)  # 2 points × 2 parameters
    assert metrics.shape == (2, 2)
    assert scalars.shape == (2,)


def test_to_numpy_without_schema_fails():
    db = Spice_Simulation_Database()
    with pytest.raises(ValueError, match="Parameter schema undefined"):
        db.to_numpy()
