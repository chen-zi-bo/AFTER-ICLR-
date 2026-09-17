import numpy as np

from models.AFTER import Model as AFTERModel
from models.ContextOffset import ContextOffsetReasoner
from models.ContextOffsetMemory import (
    EventResponseMemory,
    _history_change_shape,
    _shape_similarity,
    select_context_state_candidates,
)


class StubEmbedder:
    """Small deterministic embedder for tests that require no model weights."""

    @staticmethod
    def _encode(text):
        if "supply" in text.lower():
            return np.asarray([1.0, 0.0], dtype=float)
        return np.asarray([0.0, 1.0], dtype=float)

    def encode_documents(self, texts):
        return np.stack([self._encode(text) for text in texts])

    def encode_query(self, text):
        return self._encode(text)


def _entry(index, summary, history, future):
    return {
        "sample_index": index,
        "source_split": "train",
        "history_values": np.asarray(history, dtype=float)[:, None].tolist(),
        "true_future": np.asarray(future, dtype=float)[:, None].tolist(),
        "history_times": [f"h-{index}-{step}" for step in range(len(history))],
        "future_times": [f"f-{index}-{step}" for step in range(len(future))],
        "stage1_json": {
            "context_status": "active",
            "history_state_summary": "rising numerical history",
            "context_summary": summary,
        },
    }


def test_public_after_entry_point():
    assert AFTERModel.__name__ == "Model"


def test_only_transferable_states_enter_memory():
    stage1_results = [
        {"stage1_json": {"context_status": "none", "context_summary": "irrelevant"}},
        {"stage1_json": {"context_status": "weak", "context_summary": "possible effect"}},
        {"stage1_json": {"context_status": "active", "context_summary": "clear effect"}},
        {"stage1_json": {"context_status": "active", "context_summary": ""}},
    ]
    assert select_context_state_candidates(stage1_results) == [1, 2]


def test_shape_similarity_prefers_matching_change_pattern():
    query = _history_change_shape(np.asarray([1.0, 2.0, 4.0, 7.0]))
    matching = _history_change_shape(np.asarray([10.0, 12.0, 16.0, 22.0]))
    opposing = _history_change_shape(np.asarray([7.0, 6.0, 4.0, 1.0]))
    assert _shape_similarity(query, matching) > _shape_similarity(query, opposing)


def test_retrieval_and_elementwise_median_fusion():
    memory = EventResponseMemory(
        entries=[
            _entry(0, "supply disruption", [1.0, 2.0, 4.0], [5.0, 7.0]),
            _entry(1, "demand slowdown", [4.0, 3.0, 2.0], [1.0, 0.0]),
        ],
        embedding_model_path="unused-in-test",
        top_k=2,
        text_embedder=StubEmbedder(),
    )
    base = np.asarray([[4.5], [5.0]])
    retrieval = memory.retrieve(
        history_values=np.asarray([[1.0], [2.0], [4.0]]),
        base_forecast=base,
        stage1_json={
            "context_status": "active",
            "history_state_summary": "rising numerical history",
            "context_summary": "supply disruption",
        },
    )

    assert retrieval["examples"]
    assert retrieval["examples"][0]["memory_index"] == 0
    assert retrieval["query"]["analog_offset_center"].shape == base.shape

    context_offset = np.asarray([[0.2], [0.3]])
    memory_offset = np.asarray([[0.6], [0.9]])
    fused, metadata = ContextOffsetReasoner._fuse_offsets(
        no_memory_offsets=context_offset,
        memory_offsets=memory_offset,
        memory_retrieval=retrieval,
    )
    expected = np.median(
        np.stack(
            [context_offset, memory_offset, retrieval["query"]["analog_offset_center"]]
        ),
        axis=0,
    )
    np.testing.assert_allclose(fused, expected)
    assert metadata["used_memory"] is True
    assert metadata["method"] == "rowwise_median"
