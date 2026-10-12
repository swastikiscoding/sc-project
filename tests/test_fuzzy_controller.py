from src.fuzzy_controller import FuzzyPriorityController


def test_weights_sum_to_one():
    controller = FuzzyPriorityController()
    result = controller.predict(0.8, 0.2, 0.7)
    total = result["w_grid"] + result["w_station"] + result["w_user"]
    assert abs(total - 1.0) < 1e-8
    assert all(0 <= result[k] <= 1 for k in ("w_grid", "w_station", "w_user"))


def test_rejects_out_of_range():
    controller = FuzzyPriorityController()
    try:
        controller.predict(1.2, 0.2, 0.7)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for input outside [0, 1]")
