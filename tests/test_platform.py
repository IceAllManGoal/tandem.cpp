"""Platform helpers of tandem-run.py: hardware detection, quoting, cache dir, build state, process control.

Needs no GPU, no llama-server and no model. The Windows-only branches are exercised on any host by
flipping tandem-run.py's IS_WIN, so a Linux box still tests the Windows code paths.
Run: python -m unittest discover -s tests -v
"""
import importlib.util, json, os, shutil, signal, subprocess, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("tr", os.path.join(HERE, "..", "tandem-run.py"))
tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)

IS_WIN = os.name == "nt"
WRAPPERS = ("systemd-run", "taskset")


def fake_profile():
    return dict(gpu="RTX 4090", vram_gib=24, cores=6, ram_gib=64, pcie_gen=4)


class Platform(unittest.TestCase):
    def test_platform_flag(self):
        self.assertEqual(tr.IS_WIN, IS_WIN)

    def test_os_label(self):
        label = tr.os_label()
        self.assertTrue(label)
        if IS_WIN:
            self.assertTrue(label.startswith("Windows"), label)
        else:
            self.assertRegex(label, r"^\d+\.\d+")

    def test_hardware_detection(self):
        """The real host: /proc/meminfo + /proc/cpuinfo, or GlobalMemoryStatusEx + GetLogicalProcessorInformationEx."""
        total, lim, eff = tr.detect_ram()
        self.assertGreater(total, 0)
        self.assertGreater(eff, 0)
        self.assertLessEqual(eff, total)
        cores, name = tr.detect_cores()
        self.assertGreaterEqual(cores, 1)
        self.assertLessEqual(cores, os.cpu_count() or 1)
        self.assertTrue(name)

    def test_quoting_and_splitting(self):
        argv = ["llama-server", "--model", r"C:\models\a b.gguf"]
        if IS_WIN:
            self.assertEqual(tr.join_cmd(argv), subprocess.list2cmdline(argv))
            self.assertEqual(tr.env_prefix({"K": "v"}), '$env:K="v"; ')
            # backslashes are path separators on Windows, not escapes
            self.assertEqual(tr.split_flags(r"--model C:\models\q.gguf", win=True),
                             ["--model", r"C:\models\q.gguf"])
            self.assertEqual(tr.split_flags('--model "C:\\models\\a b.gguf"', win=True),
                             ["--model", r"C:\models\a b.gguf"])
        else:
            self.assertEqual(tr.join_cmd(argv), "llama-server --model 'C:\\models\\a b.gguf'")
            self.assertEqual(tr.env_prefix({"K": "v"}), "K=v ")
            self.assertEqual(tr.split_flags('--model "a b.gguf"', win=False), ["--model", "a b.gguf"])

    def test_cache_dir(self):
        if IS_WIN:
            base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
            self.assertEqual(tr.default_cache_dir(), os.path.join(base, "tandem"))
            self.assertTrue(os.path.isabs(tr.CACHE_DIR), tr.CACHE_DIR)
        else:
            self.assertEqual(tr.default_cache_dir(), "~/.cache/tandem")

    def test_default_server_bin(self):
        path = tr.default_server_bin()
        self.assertTrue(path)
        if os.environ.get("TANDEM_SERVER") or shutil.which("llama-server"):
            self.skipTest("TANDEM_SERVER or PATH decides the default here")
        self.assertEqual(path.lower().endswith(".exe"), IS_WIN, path)

    def test_spawn_kwargs(self):
        kwargs = tr.spawn_kwargs()
        if IS_WIN:
            self.assertEqual(kwargs.get("creationflags"), subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            self.assertTrue(kwargs.get("start_new_session"))

    def test_build_state_reads_the_marker(self):
        """libllama.so on Linux, llama.dll on Windows: the low-RAM marker decides new/old."""
        d = tempfile.mkdtemp(prefix="tandem-build-state-")
        try:
            exe = os.path.join(d, "llama-server.exe" if IS_WIN else "llama-server")
            with open(exe, "wb") as f:
                f.write(b"\0" * 64)
            lib = os.path.join(d, "llama.dll" if IS_WIN else "libllama.so")
            with open(lib, "wb") as f:
                f.write(b"\0" * 64 + tr.BUILD_STATE_NEEDLE + b"\0" * 64)
            self.assertEqual(tr.build_state(exe), "new")
            os.remove(lib)
            self.assertIn(tr.build_state(exe), ("unknown", "old"))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_windows_stop_group(self):
        """The Windows branch of stop_group(): console break to the group, then taskkill /T /F."""
        calls = []

        class Proc:
            pid = 4242
            def poll(self):
                return None
            def wait(self, timeout=None):
                raise subprocess.TimeoutExpired("llama-server", 1)

        saved = (tr.IS_WIN, tr.subprocess.run, tr.os.kill)
        tr.IS_WIN = True
        tr.subprocess.run = lambda *a, **k: calls.append(list(a[0]))
        tr.os.kill = lambda pid, sig: calls.append(("kill", pid, sig))
        try:
            tr.stop_group(Proc())
        finally:
            tr.IS_WIN, tr.subprocess.run, tr.os.kill = saved
        self.assertTrue(any(isinstance(c, list) and c[:2] == ["taskkill", "/PID"] for c in calls), calls)
        if hasattr(signal, "CTRL_BREAK_EVENT"):
            self.assertTrue(any(isinstance(c, tuple) and c[0] == "kill" for c in calls), calls)

    def test_emulated_limits(self):
        """The enforcement wrappers (systemd-run/taskset) are Linux-only; on Windows the emulation is advisory."""
        def plan_with(win):
            saved = tr.IS_WIN
            tr.IS_WIN = win
            try:
                args = tr.parse_args(["--model", "qwen3.8", "--dry-run", "--fake-hw", json.dumps(fake_profile()),
                                      "--assume-ram-gib", "16", "--assume-cores", "4", "--fake-build", "new"])
                hw, m = tr.make_plan(args)
                tr.apply_assumes(hw, args)
                plan = tr.choose_flags(hw, m, args)
                argv, _ = tr.build_command(hw, args, plan)
                return argv, plan
            finally:
                tr.IS_WIN = saved

        argv, plan = plan_with(True)
        self.assertFalse(any(os.path.basename(x) in WRAPPERS for x in argv), argv[:3])
        self.assertTrue(any("advisory" in w for w in plan.warnings), plan.warnings)

        argv, plan = plan_with(False)
        self.assertTrue(any(os.path.basename(x) in WRAPPERS for x in argv), argv[:3])
        self.assertFalse(any("advisory" in w for w in plan.warnings), plan.warnings)


if __name__ == "__main__":
    unittest.main()
