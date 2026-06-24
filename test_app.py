from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_default_resnet_flow():
    app_path = Path(__file__).resolve().parent / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=60).run()

    assert not app.exception
    assert app.button[0].label == "Запустить анализ"

    app.button[0].click().run(timeout=60)

    assert not app.exception
    markdown_values = [item.value for item in app.markdown]
    assert any("Анализ завершен" in value for value in markdown_values)
    assert any("Диагностический профиль" in value for value in markdown_values)
    assert any("NORM" in value for value in markdown_values)
    assert any("Преобладает группа NORM" in value for value in markdown_values)
