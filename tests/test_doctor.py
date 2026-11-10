"""Runtime diagnostics must distinguish missing packages from broken imports."""
from scripts import doctor


def test_import_probe_reports_missing_broken_and_importable(monkeypatch):
    monkeypatch.setattr(doctor.importlib.util, 'find_spec',
                        lambda name: None if name == 'missing' else object())

    def import_package(name):
        if name == 'broken':
            raise SystemError('incompatible binary wheel')
        return type('Module', (), {'__version__': '1.2'})()

    monkeypatch.setattr(doctor.importlib, 'import_module', import_package)
    assert doctor._probe_import('missing')['status'] == 'not_installed'
    assert doctor._probe_import('broken')['status'] == 'installed_broken'
    assert doctor._probe_import('working') == {'status': 'importable', 'version': '1.2'}


def test_invalid_application_settings_are_reported(monkeypatch):
    monkeypatch.setenv('MARKETINTEL_PORT', '70000')
    monkeypatch.setenv('MARKETINTEL_RETRIEVAL_MODE', 'semantic')
    monkeypatch.delenv('MARKETINTEL_LOCAL_EMBED_MODEL', raising=False)
    config = doctor._configuration()
    assert config['valid'] is False
    assert any('MARKETINTEL_PORT' in issue for issue in config['issues'])
    assert any('MARKETINTEL_LOCAL_EMBED_MODEL' in issue for issue in config['issues'])


def test_diagnostics_checks_mpi_extension_not_only_package(monkeypatch):
    requested = []

    def probe(name):
        requested.append(name)
        return {'status': 'importable'}

    monkeypatch.setattr(doctor, '_probe_import', probe)
    doctor.diagnostics()
    assert 'mpi4py.MPI' in requested
    assert 'mpi4py' not in requested


def test_missing_optional_parent_does_not_crash_diagnostic(monkeypatch):
    def missing_parent(name):
        raise ModuleNotFoundError("No module named 'mpi4py'")

    monkeypatch.setattr(doctor.importlib.util, 'find_spec', missing_parent)
    assert doctor._probe_import('mpi4py.MPI') == {'status': 'not_installed'}
