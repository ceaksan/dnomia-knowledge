# tests/test_git_analyze.py
from dnomia_knowledge.git_analyze import classify_crossover_results, classify_file


class TestClassifyFile:
    def test_blind(self):
        assert classify_file(churn=10, reads=0, churn_p75=5, reads_p75=5) == "BLIND"

    def test_turbulent(self):
        assert classify_file(churn=10, reads=2, churn_p75=5, reads_p75=5) == "TURBULENT"

    def test_stable(self):
        assert classify_file(churn=1, reads=10, churn_p75=5, reads_p75=5) == "STABLE"

    def test_hot(self):
        assert classify_file(churn=10, reads=10, churn_p75=5, reads_p75=5) == "HOT"

    def test_zombie(self):
        assert classify_file(churn=0, reads=3, churn_p75=5, reads_p75=5) == "ZOMBIE"

    def test_cold(self):
        assert classify_file(churn=1, reads=1, churn_p75=5, reads_p75=5) == "COLD"

    def test_zero_everything(self):
        assert classify_file(churn=0, reads=0, churn_p75=5, reads_p75=5) == "COLD"


class TestClassifyCrossoverResults:
    def test_adds_signal_column(self):
        rows = [
            {"file_path": "a.py", "churn": 10, "reads": 0, "lines_changed": 50},
            {"file_path": "b.py", "churn": 0, "reads": 5, "lines_changed": 0},
            {"file_path": "c.py", "churn": 1, "reads": 1, "lines_changed": 5},
        ]
        result = classify_crossover_results(rows)
        assert result[0]["signal"] == "BLIND"
        assert result[1]["signal"] == "ZOMBIE"

    def test_empty_input(self):
        assert classify_crossover_results([]) == []
