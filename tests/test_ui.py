from marketintel.ui import build


def test_dashboard_constructs():
    app=build()
    assert app is not None
