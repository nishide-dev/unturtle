"""
Tests for the BackboneIntegration registry (#68 PR A).

The registry centralizes model-specific loading knowledge that previously
lived as hand-written try/except blocks and a dict literal inside
``fast_diffusion_model``.  This PR is behavior-preserving, so most of these
tests pin the *existing* contract as much as the new structure.
"""

import pytest


class TestRegistryContents:
    def test_every_native_backbone_is_registered(self):
        from unturtle.models.integrations import iter_integrations

        by_type = {}
        for integration in iter_integrations():
            for model_type in integration.model_types:
                by_type[model_type] = integration

        for model_type in (
            "llada",
            "mdlm-dit",
            "dream",
            "Dream",
            "tiny-a2d-llama",
            "tiny-a2d-qwen2",
            "tiny-a2d-qwen3",
        ):
            assert model_type in by_type, f"{model_type} lost from the registry"
            assert by_type[model_type].native_model_cls is not None

    def test_dream_registers_both_casings(self):
        """DreamConfig.model_type is 'Dream'; Hub configs use both spellings."""
        from unturtle.models.integrations import resolve_native_class

        assert resolve_native_class("dream") is resolve_native_class("Dream")
        assert resolve_native_class("dream") is not None

    def test_diffusion_gemma_is_a_post_load_swap_not_a_native_class(self):
        """DiffusionGemma loads via FastModel, then gets a __class__ swap."""
        from unturtle.models.integrations import (
            resolve_native_class,
            resolve_post_load_wrapper,
        )

        assert resolve_native_class("diffusion_gemma") is None
        assert resolve_post_load_wrapper("diffusion_gemma") is not None

    def test_unknown_model_type_resolves_to_none(self):
        from unturtle.models.integrations import (
            resolve_native_class,
            resolve_post_load_wrapper,
        )

        assert resolve_native_class("not-a-real-model") is None
        assert resolve_post_load_wrapper("not-a-real-model") is None


class TestCapabilities:
    """#68 designates capabilities the extension point #69 will consume."""

    def test_declared_capabilities_are_queryable(self):
        from unturtle.models.integrations import find_integration

        llada = find_integration("llada")
        assert llada is not None
        assert llada.has_capability("masked_generation")
        assert not llada.has_capability("canvas_block_generation")

    def test_diffusion_gemma_is_canvas_not_masked(self):
        """The two generation families must not be conflated."""
        from unturtle.models.integrations import find_integration

        gemma = find_integration("diffusion_gemma")
        assert gemma is not None
        assert gemma.has_capability("canvas_block_generation")
        assert not gemma.has_capability("masked_generation")

    def test_capabilities_default_to_empty(self):
        from unturtle.models.integrations import BackboneIntegration

        bare = BackboneIntegration(name="bare", model_types=("bare",))
        assert not bare.has_capability("anything")


class TestResolverErrorHandling:
    """Only ImportError may be swallowed; anything else is a backbone bug."""

    def _integration(self, resolver):
        from unturtle.models.integrations import BackboneIntegration

        return BackboneIntegration(
            name="explosive",
            model_types=("explosive",),
            _native_resolver=resolver,
            _wrapper_resolver=resolver,
        )

    def test_import_error_is_swallowed(self):
        def boom():
            raise ImportError("optional dependency missing")

        integration = self._integration(boom)
        assert integration.native_model_cls is None
        assert integration.post_load_wrapper_cls is None

    @pytest.mark.parametrize("exc", [AttributeError, RuntimeError, TypeError])
    def test_other_exceptions_propagate(self, exc):
        """A broken backbone must fail loudly, not vanish from the map."""

        def boom():
            raise exc("backbone is broken")

        integration = self._integration(boom)
        with pytest.raises(exc):
            _ = integration.native_model_cls
        with pytest.raises(exc):
            _ = integration.post_load_wrapper_cls


class TestRegistryIsolation:
    def test_iter_integrations_cannot_mutate_the_registry(self):
        """Callers get a snapshot; appending to it must not register anything."""
        from unturtle.models.integrations import iter_integrations
        from unturtle.models.integrations import registry as reg

        before = len(reg._INTEGRATIONS)
        snapshot = iter_integrations()
        assert isinstance(snapshot, tuple)
        assert len(reg._INTEGRATIONS) == before

    def test_unregister_removes_by_identity_not_equality(self):
        """Equal-looking entries from two tests must not unregister each other."""
        from unturtle.models.integrations import (
            BackboneIntegration,
            register_integration,
        )
        from unturtle.models.integrations import registry as reg

        def resolver():
            return object

        # Field-for-field equal, so `list.remove` cannot tell them apart:
        # BackboneIntegration is a frozen dataclass with value equality.
        first = BackboneIntegration(
            name="twin", model_types=("twin",), _native_resolver=resolver
        )
        second = BackboneIntegration(
            name="twin", model_types=("twin",), _native_resolver=resolver
        )
        assert first == second and first is not second

        register_integration(first)
        # `register_integration` rejects the duplicate model_type, so append
        # directly: the hazard under test is teardown, not registration.
        reg._INTEGRATIONS.append(second)
        try:
            reg._unregister_integration(second)
            assert any(i is first for i in reg.iter_integrations()), (
                "unregistering one entry removed a different, equal-valued one"
            )
        finally:
            reg._unregister_integration(first)
            reg._unregister_integration(second)

    def test_find_integration_tolerates_none(self):
        """`config.model_type` is read defensively and may be None."""
        from unturtle.models.integrations import find_integration

        assert find_integration(None) is None


class TestLazyImports:
    def test_registry_module_itself_imports_no_backbone(self):
        """Registrations must hold resolvers, not eagerly-imported classes.

        Loaded from source in isolation rather than via ``import
        unturtle.models.integrations``: the parent ``unturtle.models`` package
        is eager and already pulls in every backbone, which would mask the
        property under test.  What matters here is that *this* module adds no
        import cost of its own, so the registry stays usable from contexts
        that must not drag in transformers' diffusion_gemma.
        """
        import pathlib
        import subprocess
        import sys

        repo_root = pathlib.Path(__file__).resolve().parents[3]
        code = (
            "import sys, importlib.util, types\n"
            "pkg = types.ModuleType('_ireg'); pkg.__path__ = "
            "['unturtle/models/integrations']\n"
            "sys.modules['_ireg'] = pkg\n"
            "for name in ('base', 'registry'):\n"
            "    spec = importlib.util.spec_from_file_location(\n"
            "        f'_ireg.{name}', f'unturtle/models/integrations/{name}.py')\n"
            "    m = importlib.util.module_from_spec(spec)\n"
            "    sys.modules[f'_ireg.{name}'] = m\n"
            "    spec.loader.exec_module(m)\n"
            "leaked = [x for x in sys.modules if 'backbones' in x "
            "or 'transformers.models.diffusion_gemma' in x]\n"
            "print(','.join(sorted(leaked)))\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=repo_root,  # the loader script uses repo-relative paths
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "", (
            f"the registry module eagerly imported: {out.stdout.strip()}"
        )

    def test_missing_backbone_drops_only_its_own_entry(self):
        """One unimportable backbone must not empty the whole map.

        Matches the old per-entry ``except ImportError: pass``.
        """
        from unturtle.models.integrations import registry as reg

        integration = next(
            i for i in reg.iter_integrations() if "llada" in i.model_types
        )
        # Save the *resolver*, not the resolved class: restoring the class
        # would leave a callable that instantiates the model on next lookup.
        original = integration._native_resolver

        def boom():
            raise ImportError("simulated missing backbone")

        object.__setattr__(integration, "_native_resolver", boom)
        try:
            classes = reg.native_model_classes()
            assert "llada" not in classes
            # Every other backbone survives.
            assert "dream" in classes
            assert "tiny-a2d-llama" in classes
        finally:
            object.__setattr__(integration, "_native_resolver", original)


class TestBackwardCompatibleShims:
    """The old module-level names are load-bearing for existing tests."""

    def test_native_model_classes_still_importable_and_shaped(self):
        from unturtle.fast_diffusion_model import _native_model_classes

        classes = _native_model_classes()
        assert isinstance(classes, dict)
        assert "llada" in classes
        assert "tiny-a2d-qwen3" in classes
        assert all(isinstance(k, str) for k in classes)

    def test_post_load_swaps_is_a_mutable_dict_of_resolvers(self):
        """`tests/models/test_diffusion_gemma.py` asserts resolver() is the class."""
        from unturtle import fast_diffusion_model as fdm
        from unturtle.models.backbones.diffusion_gemma import (
            UnturtleDiffusionGemmaForBlockDiffusion,
        )

        resolver = fdm._POST_LOAD_CLASS_SWAPS.get("diffusion_gemma")
        assert resolver is not None
        assert callable(resolver)
        assert resolver() is UnturtleDiffusionGemmaForBlockDiffusion

    def test_post_load_swaps_accepts_runtime_registration(self):
        """A test in the existing suite injects a synthetic model_type."""
        from unturtle import fast_diffusion_model as fdm

        class _Wrapper:
            pass

        class _Cfg:
            model_type = "registry-shim-test"

        class _Model:
            config = _Cfg()

        fdm._POST_LOAD_CLASS_SWAPS["registry-shim-test"] = lambda: _Wrapper
        try:
            model = _Model()
            fdm._apply_post_load_class_swap(model)
            assert type(model) is _Wrapper
        finally:
            del fdm._POST_LOAD_CLASS_SWAPS["registry-shim-test"]


class TestExtensibility:
    def test_registering_a_backbone_needs_no_central_edit(self):
        """#68's acceptance criterion, exercised rather than asserted in prose."""
        from unturtle.models.integrations import (
            BackboneIntegration,
            register_integration,
            resolve_native_class,
        )
        from unturtle.models.integrations import registry as reg

        class _Fake:
            pass

        integration = BackboneIntegration(
            name="fake-backbone",
            model_types=("fake-backbone",),
            _native_resolver=lambda: _Fake,
        )
        register_integration(integration)
        try:
            assert resolve_native_class("fake-backbone") is _Fake
            assert "fake-backbone" in reg.native_model_classes()
        finally:
            reg._unregister_integration(integration)

        assert resolve_native_class("fake-backbone") is None

    def test_duplicate_model_type_is_rejected(self):
        """Two integrations claiming one model_type is a bug, not a silent win."""
        from unturtle.models.integrations import (
            BackboneIntegration,
            register_integration,
        )

        clash = BackboneIntegration(
            name="clashing",
            model_types=("llada",),
            _native_resolver=lambda: object,
        )
        with pytest.raises(ValueError, match="llada"):
            register_integration(clash)
