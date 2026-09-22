"""探測與布林解析單元測試：共用名單解析三形狀、探測真假路徑、布林真值表，全 mock 不碰網路。"""

import types
from pathlib import Path
from unittest import mock

import ollama_shared as shared

class TestOllamaModelNames:
    def test_dict_shape(self) -> None:
        """標準 dict 回傳解析出模型名。」"""
        out = shared.ollama_model_names({"models": [{"name": "llama3.2:1b"}, {"model": "qwen3:8b"}]})
        assert out == ["llama3.2:1b", "qwen3:8b"]

    def test_list_shape(self) -> None:
        """舊版 list 回傳：字串與字典混雜皆收。」"""
        out = shared.ollama_model_names(["a", {"name": "b"}])
        assert out == ["a", "b"]

    def test_object_shape(self) -> None:
        """ollama 0.6 回傳物件（含 .models，元素取 .model／.name）。」"""
        m = types.SimpleNamespace(models=[types.SimpleNamespace(name="llama3.2:1b", model="llama3.2:1b")])
        assert shared.ollama_model_names(m) == ["llama3.2:1b"]

    def test_unknown_shape_empty(self) -> None:
        """看不懂的形狀回空串列，不拋錯。」"""
        assert shared.ollama_model_names(123) == []
        assert shared.ollama_model_names(None) == []

class TestProbeModel:
    def _client_with(self, payload: object) -> mock.Mock:
        """假 ollama client：list() 回指定形狀。」"""
        c = mock.Mock()
        c.list.return_value = payload
        return c

    def test_present_is_true(self) -> None:
        """名單有模型回 True（大小寫不敏感）。」"""
        c = self._client_with({"models": [{"name": "Llama3.2:1b"}]})
        with mock.patch.object(shared, "get_shared_client", return_value=c):
            assert shared.probe_model("llama3.2:1b", 300.0) is True

    def test_missing_is_false(self) -> None:
        """名單沒有回 False。」"""
        c = self._client_with({"models": [{"name": "qwen3:8b"}]})
        with mock.patch.object(shared, "get_shared_client", return_value=c):
            assert shared.probe_model("llama3.2:1b", 300.0) is False

    def test_error_is_false(self) -> None:
        """連線失敗與空模型名都回 False，不拋錯。」"""
        with mock.patch.object(shared, "get_shared_client", side_effect=RuntimeError("down")):
            assert shared.probe_model("llama3.2:1b", 300.0) is False
        c = self._client_with({"models": [{"name": "llama3.2:1b"}]})
        with mock.patch.object(shared, "get_shared_client", return_value=c):
            assert shared.probe_model("", 300.0) is False
            assert shared.probe_model("   ", 300.0) is False

class TestBoolEnv:
    def test_truth_table(self, monkeypatch) -> None:
        """布林真值表：多種寫法正反都認，不分大小寫。」"""
        import config as c

        for raw, want in [("1", True), ("TRUE", True), ("yes", True), ("On", True),
                          ("0", False), ("FALSE", False), ("no", False), ("Off", False)]:
            monkeypatch.setenv("DIY_BOOL_PROBE", raw)
            assert c._bool_env("DIY_BOOL_PROBE", True) is want
            assert c._bool_env("DIY_BOOL_PROBE", False) is want

    def test_unset_and_garbage_fall_back(self, monkeypatch) -> None:
        """沒設與寫壞回預設，不炸。」"""
        import config as c

        monkeypatch.delenv("DIY_BOOL_PROBE", raising=False)
        assert c._bool_env("DIY_BOOL_PROBE", True) is True
        assert c._bool_env("DIY_BOOL_PROBE", False) is False
        monkeypatch.setenv("DIY_BOOL_PROBE", "maybe")
        assert c._bool_env("DIY_BOOL_PROBE", True) is True
        assert c._bool_env("DIY_BOOL_PROBE", False) is False
