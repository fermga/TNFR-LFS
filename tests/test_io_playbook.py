from tnfr_lfs.ingestion.offline import load_playbook


def test_load_playbook_returns_rules() -> None:
    playbook = load_playbook()
    assert "delta_surplus" in playbook
    assert "sense_index_low" in playbook
    for key, actions in playbook.items():
        assert isinstance(actions, tuple)
        assert all(isinstance(entry, str) for entry in actions)
        assert all(entry.strip() for entry in actions)
