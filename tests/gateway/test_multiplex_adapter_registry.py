"""Phase 3: secondary-profile adapter registry + same-token conflict detection."""
import pytest

from gateway.run import GatewayRunner


class _FakeAdapter:
    def __init__(self, token=None, config=None):
        self.token = token
        self.config = config


class TestCredentialFingerprint:
    def test_none_without_token(self):
        assert GatewayRunner._adapter_credential_fingerprint(_FakeAdapter()) is None

    def test_stable_and_log_safe(self):
        a = _FakeAdapter(token="secret-bot-token")
        fp1 = GatewayRunner._adapter_credential_fingerprint(a)
        fp2 = GatewayRunner._adapter_credential_fingerprint(_FakeAdapter(token="secret-bot-token"))
        assert fp1 == fp2  # stable
        assert "secret-bot-token" not in (fp1 or "")  # never the raw token
        assert len(fp1) == 16

    def test_distinct_tokens_distinct_fp(self):
        a = GatewayRunner._adapter_credential_fingerprint(_FakeAdapter(token="tok-A"))
        b = GatewayRunner._adapter_credential_fingerprint(_FakeAdapter(token="tok-B"))
        assert a != b

    def test_reads_alt_attrs(self):
        class _AltAdapter:
            def __init__(self):
                self.bot_token = "alt-token"
        assert GatewayRunner._adapter_credential_fingerprint(_AltAdapter()) is not None

    def test_reads_platform_config_token(self):
        class _Config:
            token = "config-token"

        fp = GatewayRunner._adapter_credential_fingerprint(
            _FakeAdapter(token=None, config=_Config())
        )

        assert fp is not None
        assert "config-token" not in fp


    def test_reads_config_token(self):
        """Adapters like Discord store token on `config`, not on self.

        Without the config-token fallback, every Discord adapter in a
        multiplexed gateway returns None here and the same-token conflict
        check is silently skipped — N adapters start polling the same bot
        token and race on every inbound message.
        """
        class _Config:
            token = "discord-bot-token"
        class _ConfigBackedAdapter:
            config = _Config()
        fp = GatewayRunner._adapter_credential_fingerprint(_ConfigBackedAdapter())
        assert fp is not None
        assert "discord-bot-token" not in fp
        assert len(fp) == 16

    def test_distinct_config_tokens_distinct_fp(self):
        class _CfgA:
            token = "tok-A"
        class _CfgB:
            token = "tok-B"
        class _A:
            config = _CfgA()
        class _B:
            config = _CfgB()
        a = GatewayRunner._adapter_credential_fingerprint(_A())
        b = GatewayRunner._adapter_credential_fingerprint(_B())
        assert a is not None and b is not None
        assert a != b

    def test_direct_token_takes_precedence_over_config(self):
        """If both `adapter.token` and `adapter.config.token` exist, direct wins."""
        class _Cfg:
            token = "from-config"
        class _Both:
            token = "from-direct"
            config = _Cfg()
        fp = GatewayRunner._adapter_credential_fingerprint(_Both())
        import hashlib
        expected = hashlib.sha256(b"hermes-mux:from-direct").hexdigest()[:16]
        assert fp == expected

    def test_config_without_token_returns_none(self):
        """config present but no token attribute → None (no false positive)."""
        class _Cfg:
            pass
        class _Adapter:
            config = _Cfg()
        assert GatewayRunner._adapter_credential_fingerprint(_Adapter()) is None


class TestProfileMessageHandler:
    @pytest.mark.asyncio
    async def test_stamps_profile_on_unstamped_source(self):
        runner = GatewayRunner.__new__(GatewayRunner)
        seen = {}

        async def _fake_handle(event):
            seen["profile"] = event.source.profile
            return "ok"

        runner._handle_message = _fake_handle
        handler = runner._make_profile_message_handler("coder")

        class _Src:
            profile = None

        class _Evt:
            source = _Src()

        result = await handler(_Evt())
        assert result == "ok"
        assert seen["profile"] == "coder"

    @pytest.mark.asyncio
    async def test_does_not_override_existing_profile(self):
        runner = GatewayRunner.__new__(GatewayRunner)
        seen = {}

        async def _fake_handle(event):
            seen["profile"] = event.source.profile
            return "ok"

        runner._handle_message = _fake_handle
        handler = runner._make_profile_message_handler("coder")

        class _Src:
            profile = "writer"  # already stamped (e.g. by URL prefix)

        class _Evt:
            source = _Src()

        await handler(_Evt())
        assert seen["profile"] == "writer"


class TestPortBindingHardError:
    """A secondary profile enabling a port-binding platform aborts startup."""

    @pytest.mark.asyncio
    async def test_secondary_webhook_raises(self, monkeypatch):
        from gateway.run import MultiplexConfigError
        from gateway.config import GatewayConfig, Platform, PlatformConfig

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner._profile_adapters = {}

        # reviewer profile config enables webhook (a port-binding platform)
        reviewer_cfg = GatewayConfig(multiplex_profiles=True)
        reviewer_cfg.platforms = {
            Platform.WEBHOOK: PlatformConfig(enabled=True, extra={"port": 8644}),
        }
        monkeypatch.setattr(
            "gateway.config.load_gateway_config", lambda: reviewer_cfg
        )

        with pytest.raises(MultiplexConfigError) as ei:
            await runner._start_one_profile_adapters("reviewer", "/tmp/x", {})
        assert "webhook" in str(ei.value)
        assert "reviewer" in str(ei.value)

    @pytest.mark.asyncio
    async def test_secondary_non_binding_platform_ok(self, monkeypatch):
        """A non-port-binding platform (e.g. telegram) is NOT rejected."""
        from gateway.config import GatewayConfig, Platform, PlatformConfig

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner._profile_adapters = {}

        reviewer_cfg = GatewayConfig(multiplex_profiles=True)
        reviewer_cfg.platforms = {
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="t"),
        }
        monkeypatch.setattr(
            "gateway.config.load_gateway_config", lambda: reviewer_cfg
        )
        # _create_adapter returns None here (no real telegram token wiring), so
        # the loop simply connects nothing — the key assertion is NO raise.
        monkeypatch.setattr(runner, "_create_adapter", lambda p, c: None)

        connected = await runner._start_one_profile_adapters("reviewer", "/tmp/x", {})
        assert connected == 0  # nothing connected, but no MultiplexConfigError

    @pytest.mark.asyncio
    async def test_secondary_same_config_token_is_refused(self, monkeypatch):
        """Adapters that keep their token on config still trip the mux guard."""
        from gateway.config import GatewayConfig, Platform, PlatformConfig

        class _ConfigTokenAdapter:
            def __init__(self, token):
                self.config = PlatformConfig(enabled=True, token=token)
                self.disconnected = False

            async def connect(self):
                raise AssertionError("duplicate adapter must not connect")

            async def disconnect(self):
                self.disconnected = True

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner._profile_adapters = {}

        reviewer_cfg = GatewayConfig(multiplex_profiles=True)
        reviewer_cfg.platforms = {
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="same-token"),
        }
        duplicate = _ConfigTokenAdapter("same-token")
        claimed = {
            (
                Platform.TELEGRAM,
                GatewayRunner._adapter_credential_fingerprint(
                    _ConfigTokenAdapter("same-token")
                ),
            ): "default"
        }

        monkeypatch.setattr(
            "gateway.config.load_gateway_config", lambda: reviewer_cfg
        )
        monkeypatch.setattr(runner, "_create_adapter", lambda p, c: duplicate)
        monkeypatch.setattr(runner, "_adapter_disconnect_timeout_secs", lambda: 0)

        connected = await runner._start_one_profile_adapters(
            "reviewer", "/tmp/x", claimed
        )

        assert connected == 0
        assert duplicate.disconnected is True
        assert runner._profile_adapters["reviewer"] == {}

    def test_port_binding_set_covers_known_listeners(self):
        from gateway.run import _PORT_BINDING_PLATFORM_VALUES
        # Every adapter that binds a TCP port must be in the guard set.
        for p in ("webhook", "api_server", "msgraph_webhook", "feishu",
                  "wecom_callback", "bluebubbles", "sms"):
            assert p in _PORT_BINDING_PLATFORM_VALUES



class TestOutboundAdapterForSource:
    """_adapter_for_source must route a reply back through the SAME account
    the inbound arrived on. A profile-stamped source resolves to that
    profile's adapter (its own credential/number), not the default map."""

    def _runner(self):
        from gateway.config import Platform

        runner = GatewayRunner.__new__(GatewayRunner)
        default_signal = _FakeAdapter(token="default-number")
        profile_signal = _FakeAdapter(token="equity-number")
        runner.adapters = {Platform.SIGNAL: default_signal}
        runner._profile_adapters = {
            "equity-analyst": {Platform.SIGNAL: profile_signal},
        }
        return runner, default_signal, profile_signal

    class _Src:
        def __init__(self, platform, profile=None):
            self.platform = platform
            self.profile = profile

    def test_stamped_source_uses_profile_adapter(self):
        from gateway.config import Platform

        runner, default_signal, profile_signal = self._runner()
        src = self._Src(Platform.SIGNAL, profile="equity-analyst")
        assert runner._adapter_for_source(src) is profile_signal
        assert runner._adapter_for_source(src) is not default_signal

    def test_unstamped_source_uses_default_adapter(self):
        from gateway.config import Platform

        runner, default_signal, _ = self._runner()
        src = self._Src(Platform.SIGNAL, profile=None)
        assert runner._adapter_for_source(src) is default_signal

    def test_default_profile_stamp_falls_back_to_default(self):
        from gateway.config import Platform

        runner, default_signal, _ = self._runner()
        # A profile with no per-profile adapter for this platform (e.g. the
        # active/default profile) falls back to self.adapters, unchanged.
        src = self._Src(Platform.SIGNAL, profile="default")
        assert runner._adapter_for_source(src) is default_signal

    def test_none_source_returns_none(self):
        runner, _, _ = self._runner()
        assert runner._adapter_for_source(None) is None

    def test_missing_adapters_attr_returns_none(self):
        from gateway.config import Platform

        runner = GatewayRunner.__new__(GatewayRunner)
        src = self._Src(Platform.SIGNAL, profile=None)
        assert runner._adapter_for_source(src) is None

    def test_served_secondary_missing_platform_adapter_returns_none(self):
        """P1 (#79 review): a served secondary profile whose adapter for this
        platform is absent (e.g. its Signal adapter failed to connect) must
        NOT fall back to the default adapter — that would leak a reply out the
        default account. It returns None so the caller defers/drops instead."""
        from gateway.config import Platform

        runner, default_signal, _ = self._runner()
        # equity-analyst is a served secondary (has a _profile_adapters entry)
        # but has no adapter for TELEGRAM.
        src = self._Src(Platform.TELEGRAM, profile="equity-analyst")
        assert runner._adapter_for_source(src) is None
        assert runner._adapter_for_source(src) is not default_signal

    def test_served_secondary_with_empty_adapter_map_returns_none(self):
        """A secondary profile whose every adapter failed to connect keeps an
        empty _profile_adapters entry (setdefault seeds it). It must still
        resolve to None, never the default account."""
        from gateway.config import Platform

        runner, default_signal, _ = self._runner()
        runner._profile_adapters["research"] = {}  # all adapters failed
        src = self._Src(Platform.SIGNAL, profile="research")
        assert runner._adapter_for_source(src) is None
        assert runner._adapter_for_source(src) is not default_signal

    def test_shared_listener_platform_falls_back_to_default(self):
        """P2 (#79 review): shared listener platforms (webhook/api_server/…) are
        default-owned — a secondary profile can never bind its own, and the
        single default adapter serves every profile via /p/<profile>/. So a
        profile-stamped source on such a platform must fall back to the shared
        default adapter, NOT be dropped as a 'missing secondary adapter'."""
        from gateway.config import Platform

        runner = GatewayRunner.__new__(GatewayRunner)
        default_webhook = _FakeAdapter(token="default-webhook")
        runner.adapters = {Platform.WEBHOOK: default_webhook}
        # equity-analyst is a served secondary but (correctly) owns no webhook
        # adapter — port-binding platforms are default-owned.
        runner._profile_adapters = {"equity-analyst": {}}

        src = self._Src(Platform.WEBHOOK, profile="equity-analyst")
        assert runner._adapter_for_source(src) is default_webhook

    def test_removed_secondary_stamp_returns_none(self):
        """P2 (#79 follow-up): a stamp naming a non-default profile that is NOT
        a served secondary (removed/renamed, or its startup failed before
        registering in _profile_adapters) has no correct account. It must
        resolve to None — never fall back to the default account and leak the
        reply/restore notice out the operator's default credential."""
        from gateway.config import Platform

        runner, default_signal, _ = self._runner()
        runner._owner_profile_name = "default"
        # 'ghost' is not in _profile_adapters and is not the owner profile.
        src = self._Src(Platform.SIGNAL, profile="ghost")
        assert runner._adapter_for_source(src) is None
        assert runner._adapter_for_source(src) is not default_signal

    def test_removed_stamp_dropped_even_under_scoped_active_profile(self):
        """Regression for the Codex P1 / claude blocker: on the primary path
        _adapter_for_source runs inside _profile_runtime_scope(source.profile),
        so a LIVE _active_profile_name() would echo back the stamped profile and
        the drop would never fire. The guard compares against the startup-captured
        owner name instead, so a removed 'ghost' stamp is still dropped even when
        the (scoped) active-profile call returns 'ghost'."""
        from gateway.config import Platform

        runner, default_signal, _ = self._runner()
        runner._owner_profile_name = "default"
        # Simulate the in-turn scope: a live active-profile call returns the stamp.
        runner._active_profile_name = lambda: "ghost"
        src = self._Src(Platform.SIGNAL, profile="ghost")
        assert runner._adapter_for_source(src) is None
        assert runner._adapter_for_source(src) is not default_signal

    def test_active_profile_stamp_falls_back_to_default(self):
        """A stamp naming the owner profile (whose adapters live in
        self.adapters and which is intentionally never in _profile_adapters)
        must still fall back to the default adapter — dropping it would break
        delivery for the gateway's own home profile."""
        from gateway.config import Platform

        runner, default_signal, _ = self._runner()
        runner._owner_profile_name = "reviewer"
        src = self._Src(Platform.SIGNAL, profile="reviewer")
        assert runner._adapter_for_source(src) is default_signal

    def test_removed_secondary_stamp_on_shared_listener_falls_back(self):
        """A shared-listener platform is default-owned regardless of the stamp:
        even an unknown/removed profile stamp resolves to the single default
        adapter (the /p/<profile>/ prefix routes it), never dropped."""
        from gateway.config import Platform

        runner = GatewayRunner.__new__(GatewayRunner)
        default_webhook = _FakeAdapter(token="default-webhook")
        runner.adapters = {Platform.WEBHOOK: default_webhook}
        runner._profile_adapters = {}
        runner._owner_profile_name = "default"
        src = self._Src(Platform.WEBHOOK, profile="ghost")
        assert runner._adapter_for_source(src) is default_webhook



