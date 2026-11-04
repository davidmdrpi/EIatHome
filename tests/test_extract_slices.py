"""tools/extract_slices.py: remote range reads, coarse-channel selection and the drift search."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import re
import struct
import tempfile
import threading
import unittest
from unittest import mock
import urllib.parse

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("extract_slices", ROOT / "tools/extract_slices.py")
x = importlib.util.module_from_spec(spec)
spec.loader.exec_module(x)


def write_h5(path, data, fch1=8000.0, foff=-1e-6, tsamp=6.0, chunk=None, time_chunk=1):
    with h5py.File(path, "w") as f:
        dset = f.create_dataset("data", data=data[:, None, :], chunks=(time_chunk, 1, chunk or data.shape[1]), compression="gzip")
        dset.attrs.update(fch1=fch1, foff=foff, tsamp=tsamp, tstart=60000.25, nchans=data.shape[1], source_name=b"TEST")


import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
import skycoords  # noqa: E402

ON_2020 = {"src_raj": 17.211244722222222, "src_dej": 12.403781666666667, "tstart": 59046.92634259259,
           "source_name": "VOYAGER-1", "telescope_id": 6}


class RangeServer:
    """Serves one file over HTTP, honouring Range requests unless told to ignore them."""

    def __init__(self, path, honour_range=True):
        payload = Path(path).read_bytes()

        class Handler(BaseHTTPRequestHandler):
            def do_HEAD(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()

            def do_GET(self):
                match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
                body, status = payload, 200
                if match and honour_range:
                    body, status = payload[int(match[1]):int(match[2]) + 1], 206
                self.send_response(status)
                if status == 206:
                    self.send_header("Content-Range", f"bytes {match[1]}-{match[2]}/{len(payload)}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/file.h5"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class ExtractorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def serve(self, path, **kw):
        server = RangeServer(path, **kw)
        self.addCleanup(server.close)
        return server.url

    def test_drift_grid_keeps_the_published_16_step_search(self):
        # The published 16-frame slices were cut with this grid; regenerating them must not change it.
        np.testing.assert_array_equal(x.drift_trials(16), np.arange(-4, 4.01, 0.25))
        for n_t in [16, 49, 279]:
            trials = x.drift_trials(n_t)
            self.assertAlmostEqual((trials[1] - trials[0]) * n_t, 4)  # at most 2 channels off over the scan
            np.testing.assert_allclose([trials[0], trials[-1]], [-4, 4], rtol=0, atol=1e-9)

    def test_range_reads_fetch_each_chunk_of_one_coarse_channel_once(self):
        rng = np.random.default_rng(1)
        width, coarse = 4096, 8
        data = (100 + rng.normal(size=(5, coarse * width))).astype(np.float32)
        freq = 8000.0 - 1e-6 * (5 * width + 10)  # inside coarse channel 5
        original = x.RangeFile._get
        # Chunks of one time step (as in Breakthrough Listen files) and spanning all five.
        for time_chunk in [1, 5]:
            with self.subTest(time_chunk=time_chunk):
                path = self.folder / f"node-{time_chunk}.h5"
                write_h5(path, data, chunk=width, time_chunk=time_chunk)
                url = self.serve(path)
                with h5py.File(path, "r") as f:
                    dset = f["data"]
                    selected = {(c.byte_offset, c.size) for c in (dset.id.get_chunk_info_by_coord((t, 0, 5 * width))
                                                                  for t in range(0, 5, time_chunk))}
                fetches = Counter()

                def counting(self, offset, length):
                    fetches[(offset, length)] += 1
                    return original(self, offset, length)

                with mock.patch.object(x, "COARSE_WIDTH", width), mock.patch.object(x.RangeFile, "WHOLE", 0):
                    local, local_header = x.load(path, freq)
                    with mock.patch.object(x.RangeFile, "_get", counting):
                        remote, remote_header = x.load(url, freq)
                np.testing.assert_array_equal(local, data[:, 5 * width:6 * width])
                np.testing.assert_array_equal(remote, local)
                self.assertEqual(local_header, remote_header)
                self.assertEqual(local_header["coarse_channel"], 5)
                self.assertEqual(local_header["nchans"], width)
                self.assertAlmostEqual(local_header["fch1"], 8000.0 - 1e-6 * 5 * width)
                # Each chunk of that coarse channel is downloaded exactly once; nothing else that
                # large is, so the rest of the file stays on the server.
                self.assertEqual({r: fetches[r] for r in selected}, {r: 1 for r in selected})
                self.assertEqual({r for r in fetches if r[1] > x.RangeFile.BLOCK}, {r for r in selected if r[1] > x.RangeFile.BLOCK})
                self.assertLess(sum(length * n for (_, length), n in fetches.items()), path.stat().st_size / 2)

    def test_a_server_that_ignores_range_is_refused_not_downloaded(self):
        data = np.ones((2, 64), dtype=np.float32)
        path = self.folder / "small.h5"
        write_h5(path, data)
        with self.assertRaisesRegex(OSError, "doesn't support range requests"):
            x.RangeFile(self.serve(path, honour_range=False))

    def test_a_small_file_is_fetched_in_one_request(self):
        data = np.arange(2 * 64, dtype=np.float32).reshape(2, 64)
        path = self.folder / "small.h5"
        write_h5(path, data)
        remote = x.RangeFile(self.serve(path))
        self.assertEqual(remote.fetched, path.stat().st_size, "the whole file, once")
        self.assertEqual(remote.read(), path.read_bytes())
        self.assertEqual(remote.fetched, path.stat().st_size, "and nothing more")

    def test_inputs_and_frequencies_outside_the_file(self):
        self.assertEqual(x.input_path("https://example.org/a/b.h5"), "https://example.org/a/b.h5")
        self.assertEqual(x.name_of("https://example.org/a/b.h5?x=1"), "b.h5")
        self.assertEqual(x.input_path("data/a.h5"), Path("data/a.h5"))
        with self.assertRaises(argparse.ArgumentTypeError):
            x.input_path("http://blpd0.ssl.berkeley.edu/Voyager_data/Voyager1.single_coarse.fine_res.h5")
        path = self.folder / "two.h5"
        write_h5(path, np.ones((2, 128), dtype=np.float32))
        with mock.patch.object(x, "COARSE_WIDTH", 64):
            for freq in [8000.0 + 1e-6, 8000.0 - 1e-6 * 200]:  # above the first channel; past the last
                with self.assertRaises(SystemExit):
                    x.load(path, freq)

    def test_49_step_search_finds_a_line_between_the_old_grid_points(self):
        # A line drifting 0.875 channels/step (between the old 0.75 and 1.0 trials) and a fainter
        # one at exactly 1.0. On a 0.25 grid the true line smears over 6 channels in 49 steps
        # and the decoy wins; on the scaled grid the brighter line is found.
        rng = np.random.default_rng(3)
        n_t, n_chan = 49, 8192
        data = (100 + rng.normal(size=(n_t, n_chan))).astype(np.float32)
        steps = np.arange(n_t)
        data[steps, np.round(1000 + 0.875 * steps).astype(int)] += 6
        data[steps, np.round(6000 + 1.0 * steps).astype(int)] += 4
        path = self.folder / "lines.h5"
        write_h5(path, data)
        args = argparse.Namespace(width=256, max_signals=1, threshold=15, prefix="test", source_base_url=None,
                                  product=None, coarse_channel_at=None)
        [(_, entry)] = x.extract(path, args)
        self.assertAlmostEqual(entry["source"]["drift_channels_per_step"], 0.875, delta=0.03)
        self.assertLess(abs(entry["source"]["first_channel"] + 128 - (1000 + 0.875 * n_t / 2)), 3)
        with mock.patch.object(x, "drift_trials", lambda n_t: np.arange(-4, 4.01, 0.25)):
            [(_, old)] = x.extract(path, args)
        self.assertGreater(old["source"]["first_channel"], 5000, "the fixed 0.25 grid picks the decoy")


def write_fil(path, data, **header):
    """A SIGPROC filterbank file: keyword header, then float32 samples, time step by time step."""
    def string(text):
        return struct.pack("<i", len(text)) + text.encode()
    kinds = {"telescope_id": "i", "nbits": "i", "nchans": "i", "nifs": "i", "source_name": "s"}
    out = string("HEADER_START")
    for key, value in {"nbits": 32, "nifs": 1, "nchans": data.shape[1], **header}.items():
        kind = kinds.get(key, "d")
        out += string(key) + (string(value) if kind == "s" else struct.pack(f"<{kind}", value))
    Path(path).write_bytes(out + string("HEADER_END") + data.astype("<f4").tobytes())


class FileServer:
    """Serves several files and JSON replies over HTTP, honouring Range requests."""

    def __init__(self, files):
        self.files = files

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                payload = files.get(self.path)
                if payload is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                if callable(payload):
                    payload = payload(self.path)
                match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
                body, status = payload, 200
                if match:
                    body, status = payload[int(match[1]):int(match[2]) + 1], 206
                self.send_response(status)
                if status == 206:
                    self.send_header("Content-Range", f"bytes {match[1]}-{match[2]}/{len(payload)}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class PoolTests(unittest.TestCase):
    """--pool: the archive's query-files API, on-target hits, blind windows and the pool's limits."""

    WIDTH = 4096  # fine channels per coarse channel in these small test files

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        x._remotes.clear()

    def scan(self, name, tstart, line_at=None, n_t=16, seed=0, dc=True):
        """A two-coarse-channel scan of TRAPPIST-1 (nfpc in the header gives the channelization),
        with a DC spike in the middle of each coarse channel and maybe a drifting line."""
        rng = np.random.default_rng(seed)
        data = (100 + rng.normal(size=(n_t, 2 * self.WIDTH))).astype(np.float32)
        if dc:
            data[:, self.WIDTH // 2::self.WIDTH] *= 20
        if line_at is not None:
            steps = np.arange(n_t)
            data[steps, np.round(line_at + 0.5 * steps).astype(int)] += 8
        path = self.folder / name
        with h5py.File(path, "w") as f:
            dset = f.create_dataset("data", data=data[:, None, :], chunks=(1, 1, 1024))
            dset.attrs.update(fch1=1500.0, foff=-2.7939677238464355e-06, tsamp=18.253611008, tstart=tstart,
                              nchans=data.shape[1], source_name=b"TRAPPIST1", telescope_id=6, nfpc=self.WIDTH,
                              src_raj=23.108133, src_dej=-5.041389)
        return path.read_bytes()

    def run_pool(self, targets, out=None, allow_http=True, limits=None, only=None):
        pool = {"id": "test-pool/v1", "seed": 7, "api": f"{self.server.base}/api", "threshold": 10,
                "limits": limits or {}, "targets": targets}
        pool_path = self.folder / "pool.json"
        pool_path.write_text(json.dumps(pool))
        out = out or self.folder / "slices"
        out.mkdir(exist_ok=True)
        args = argparse.Namespace(pool=pool_path, out=out, only=only, allow_http=allow_http, width=256,
                                  product=None, source_base_url=None)
        log = io.StringIO()
        try:
            with redirect_stdout(log):
                x.run_pool(args)
        finally:
            self.last_log = log.getvalue()
        return json.loads((out / "index.json").read_text())["slices"]

    def archive(self, name, mjd, **extra):
        return {"target": "TRAPPIST1", "telescope": "GBT", "utc": "2017-03-01 12:00:00", "mjd": mjd,
                "ra": 346.62224, "decl": -5.0413, "center_freq": 1500.0, "file_type": "HDF5", "size": 1,
                "md5sum": "0" * 32, "url": f"{self.server.base}/dl/{name}", "cadence_url": None, **extra}

    def setup_archive(self, scans, listing_for):
        """An archive serving these scans; listing_for(query) gives query-files's data."""
        files = {f"/dl/{name}": payload for name, payload in scans.items()}
        queries = []

        class Files(dict):
            def get(inner, path, default=None):
                if path.startswith("/api/query-files"):
                    query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(path).query))
                    queries.append(query)
                    return json.dumps({"result": "success", "data": listing_for(query)}).encode()
                return dict.get(inner, path, default)

        self.server = FileServer(Files(files))
        self.addCleanup(self.server.close)
        return queries

    def test_a_target_gives_hits_and_blind_windows_with_its_sky_and_archive_record(self):
        scans = {"a.0000.h5": self.scan("a.0000.h5", 57800.25, line_at=self.WIDTH + 900),
                 "b.0000.h5": self.scan("b.0000.h5", 57801.25, line_at=self.WIDTH + 2900, seed=1),
                 "c.0002.h5": self.scan("c.0002.h5", 57802.25, seed=2)}  # a mid-resolution product: never chosen
        queries = self.setup_archive(scans, lambda q: [] if q["target"] == "NOPE" else [
            self.archive("a.0000.h5", 57800.25), self.archive("b.0000.h5", 57801.25), self.archive("c.0002.h5", 57802.25)])
        target = {"key": "trappist1", "name": "TRAPPIST-1", "distance_label": "40.7 light-years",
                  "queries": [{"target": "NOPE"}, {"target": "TRAPPIST"}], "files": 2, "hits": 1, "blind": 2, "tries": 1}
        # Every coarse channel drawn must be the one with the line: give each file one choice.
        with mock.patch.object(x, "coarse_frequencies", lambda header, n, t, path, seed: [1500.0 - 2.7939677238464355e-06 * (1.5 * self.WIDTH)]):
            slices = self.run_pool([target])
        self.assertEqual([q["target"] for q in queries], ["NOPE", "TRAPPIST"], "the first query that finds files wins")
        by_family = Counter(e["family"] for e in slices)
        self.assertEqual(by_family, {"stars": 2, "blind-window": 4})
        for e in slices:
            src = e["source"]
            self.assertEqual(src["pool"], {"id": "test-pool/v1", "key": "trappist1"})
            self.assertEqual(src["sky"]["target"], "TRAPPIST-1")
            self.assertEqual(src["sky"]["distance_label"], "40.7 light-years")
            self.assertEqual(src["sky"]["telescope"], "Green Bank Telescope")
            self.assertEqual(src["sky"]["constellation"], "Aquarius")
            self.assertAlmostEqual(src["sky"]["ra_hours"], 23.108133, places=5)
            self.assertIn(src["archive"]["url"].rsplit("/", 1)[1], {"a.0000.h5", "b.0000.h5"})
            self.assertEqual(src["archive"]["md5sum"], "0" * 32)
            self.assertEqual(src["coarse_channel"], 1)
            self.assertEqual(e["shape"], [16, 256])
            self.assertTrue((self.folder / "slices" / e["file"]).exists())
        hits = sorted((e for e in slices if e["family"] == "stars"), key=lambda e: e["source"]["tstart_mjd"])
        for hit, line_at in zip(hits, [900, 2900]):
            self.assertEqual(hit["group"], "hit")
            self.assertAlmostEqual(hit["source"]["drift_channels_per_step"], 0.5, delta=0.05)
            self.assertLess(abs(hit["source"]["first_channel"] + 128 - (line_at + 0.5 * 8)), 3)
            self.assertGreater(hit["source"]["snr"], 10)
        for blind in (e for e in slices if e["family"] == "blind-window"):
            self.assertEqual(blind["group"], "blind window")
            first = blind["source"]["first_channel"]
            self.assertTrue(204 <= first and first + 256 <= self.WIDTH - 204, "away from the band edges")
            self.assertFalse(self.WIDTH // 2 - 256 < first + 256 and first < self.WIDTH // 2 + 128, "and the DC spike")

    def test_rerunning_the_pool_replaces_its_slices_and_keeps_the_rest(self):
        scans = {"a.0000.h5": self.scan("a.0000.h5", 57800.25, line_at=900)}
        self.setup_archive(scans, lambda q: [self.archive("a.0000.h5", 57800.25)])
        out = self.folder / "slices"
        out.mkdir()
        voyager = {"id": "voyager1-1-carrier", "family": "voyager", "group": "carrier", "file": "v.bin", "sha256": "f" * 64,
                   "source": {"tstart_mjd": 1.0}}
        (out / "v.bin").write_bytes(b"v")
        (out / "index.json").write_text(json.dumps({"format": "eiathome-slices", "version": 2, "slices": [voyager]}))
        target = {"key": "trappist1", "name": "TRAPPIST-1", "queries": [{"target": "TRAPPIST"}], "files": 1, "hits": 1, "blind": 3}
        first = self.run_pool([target], out=out)
        files = {e["file"]: (out / e["file"]).read_bytes() for e in first}
        again = self.run_pool([target], out=out)
        self.assertEqual(first, again, "the seed fixes every choice")
        self.assertEqual(files, {e["file"]: (out / e["file"]).read_bytes() for e in again})
        self.assertIn(voyager, again)
        self.assertEqual(sum(e["family"] == "blind-window" for e in again), 3)
        # Fewer blind windows now: the old ones go, files and all.
        fewer = self.run_pool([{**target, "blind": 1}], out=out)
        self.assertEqual(sum(e["family"] == "blind-window" for e in fewer), 1)
        self.assertEqual(sorted(p.name for p in out.glob("*.bin")), sorted(e["file"] for e in fewer))

    def test_the_pool_stops_at_its_limits(self):
        scans = {f"{n}.0000.h5": self.scan(f"{n}.0000.h5", 57800.25 + n, seed=n) for n in range(3)}
        self.setup_archive(scans, lambda q: [self.archive(f"{n}.0000.h5", 57800.25 + n) for n in range(3)])
        target = {"key": "trappist1", "name": "TRAPPIST-1", "queries": [{"target": "TRAPPIST"}], "files": 3, "hits": 0, "blind": 2}
        self.assertEqual(len(self.run_pool([target], limits={"max_slices": 5})), 5)
        self.assertEqual(len(self.run_pool([target], limits={"max_bytes": 3 * 16 * 256 * 4})), 3)
        two = [{**target, "key": f"star{n}", "name": f"Star {n}"} for n in range(3)]
        slices = self.run_pool(two, limits={"max_targets": 2})
        self.assertEqual({e["source"]["pool"]["key"] for e in slices}, {"star0", "star1"})

    def test_a_file_that_fails_is_replaced_and_off_source_scans_are_never_on_target(self):
        scans = {"short.0000.h5": self.scan("short.0000.h5", 57800.25, n_t=3),
                 "nodc.0000.h5": self.scan("nodc.0000.h5", 57801.25, line_at=900, dc=False),
                 "off.0000.h5": self.scan("off.0000.h5", 57802.25, line_at=900),
                 "good.0000.h5": self.scan("good.0000.h5", 57803.25, line_at=900)}
        self.setup_archive(scans, lambda q: [
            self.archive("short.0000.h5", 57800.25), self.archive("nodc.0000.h5", 57801.25),
            {**self.archive("off.0000.h5", 57802.25), "target": "TRAPPIST1_OFF"}, self.archive("good.0000.h5", 57803.25)])
        target = {"key": "trappist1", "name": "TRAPPIST-1", "queries": [{"target": "TRAPPIST"}], "files": 1, "hits": 1, "blind": 1}
        # The seeded order of the files (the server's port changes the URLs): good last.
        order = {"short": 0.1, "nodc": 0.2, "off": 0.3, "good": 0.4}
        draw = x.draw
        def ordered(seed, *parts):
            name = str(parts[-1]).rsplit("/", 1)[-1].split(".")[0] if len(parts) == 2 else None
            return order[name] if name in order else draw(seed, *parts)
        with mock.patch.object(x, "draw", ordered):
            slices = self.run_pool([target])
        self.assertEqual({e["source"]["archive"]["url"].rsplit("/", 1)[1] for e in slices}, {"good.0000.h5"})
        self.assertIn("has 3 time steps", self.last_log)
        self.assertIn("no DC spike in the middle", self.last_log)
        self.assertNotIn("off.0000.h5", self.last_log, "an off-source scan is never read as on-target")

    def test_parkes_coarse_channels_are_tried_on_both_alignments(self):
        # A Parkes-like filterbank file (telescope 4, 1 MHz coarse channels of 2048 fine ones
        # here) whose DC spikes sit in the middle of the fch1-aligned blocks, not on whole MHz.
        n = 2048
        rng = np.random.default_rng(5)
        data = (100 + rng.normal(size=(16, 4 * n))).astype(np.float32)
        data[:, n // 2::n] *= 20
        steps = np.arange(16)
        data[steps, np.round(2 * n + 300 + 0.5 * steps).astype(int)] += 8
        fil = self.folder / "pks.fil"
        foff = 1 / n
        write_fil(fil, data, telescope_id=4, source_name="PROXCEN", fch1=980.0, foff=foff, tsamp=16.78,
                  tstart=58602.5, src_raj=142942.95, src_dej=-624046.2)
        self.setup_archive({"pks.fil": fil.read_bytes()}, lambda q: [{**self.archive("pks.fil", 58602.5), "telescope": "Parkes", "file_type": "filterbank"}])
        target = {"key": "proxcen", "name": "Proxima Centauri", "queries": [{"target": "PROXCEN"}], "files": 1, "hits": 1,
                  "blind": 1, "freq_mhz": 980.0 + foff * (2 * n + 600)}
        # The test file's channels are 0.5 kHz wide, and its band too narrow for the edge masks.
        with mock.patch.object(x, "searchable", lambda snr, width: snr), mock.patch.object(x, "FINE_HZ", 1e4):
            slices = self.run_pool([target])
        self.assertIn("trying coarse channels offset by 0.0", self.last_log)
        hit = next(e for e in slices if e["family"] == "stars")
        self.assertEqual(hit["source"]["sky"]["telescope"], "Parkes")
        self.assertEqual(hit["source"]["coarse_channel"], 2)
        self.assertAlmostEqual(hit["source"]["fch1_mhz"], 980.0 + foff * 2 * n)
        self.assertLess(abs(hit["source"]["first_channel"] + 128 - (300 + 0.5 * 8)), 3)

    def test_fine_resolution_products_by_name(self):
        def ok(name, kind="HDF5"):
            return x.fine_resolution({"url": f"http://h/{name}", "file_type": kind})
        self.assertTrue(ok("spliced_blc0001020304050607_guppi_57532_02535_GJ406_0007.gpuspec.0000.h5"))
        self.assertTrue(ok("Parkes_58031_47052_HIP8102_S_fine.h5"))
        self.assertTrue(ok("blc01_ProxCen_S_2019-04-29T12:56:23.000.fil", "filterbank"), "unmarked: the header decides")
        self.assertFalse(ok("blc00_guppi_57807_78936_DIAG_TRAPPIST1_0032.gpuspec.0001.fil", "filterbank"))
        self.assertFalse(ok("spliced_blc0001020304050607_guppi_57532_02535_GJ406_0007.gpuspec.0002.h5"))
        self.assertFalse(ok("spliced_blc02030405_2bit_guppi_57457_48661_GJ699_0005.gpuspec.0000.h5"), "2-bit samples don't normalize")
        self.assertFalse(ok("guppi_59337_61382_ProxCen_S_0001.0000.raw", "baseband data"))

    def test_the_archive_names_the_telescope_when_the_header_does_not(self):
        header = {"telescope_id": 64, "foff": 2.0e-06}
        self.assertEqual(x.channels_per_coarse(header), x.COARSE_WIDTH, "an id this tool doesn't know")
        x.apply_hints(header, {"telescope_id": 4})
        self.assertEqual(x.channels_per_coarse(header), 500_000)
        gbt = {"telescope_id": 6, "foff": -2.7939677238464355e-06}
        self.assertEqual(x.apply_hints(dict(gbt), {"telescope_id": 4}), gbt, "a header that names a known telescope wins")

    def test_files_outside_the_targets_band_are_passed_over_without_counting(self):
        scans = {f"{n}.0000.h5": self.scan(f"{n}.0000.h5", 57800.25 + n, line_at=900, seed=n) for n in range(5)}
        self.setup_archive(scans, lambda q: [self.archive(f"{n}.0000.h5", 57800.25 + n) for n in range(5)])
        # The scans cover 1500 MHz down by 2 x 4096 channels; this frequency is in none of them.
        target = {"key": "trappist1", "name": "TRAPPIST-1", "queries": [{"target": "TRAPPIST"}], "files": 1, "hits": 1,
                  "blind": 1, "freq_mhz": 982.0}
        with self.assertRaisesRegex(SystemExit, "no slices"):
            self.run_pool([target])
        self.assertEqual(self.last_log.count("passed over"), 5, "every candidate is looked at, none counts as a try")

    def test_plain_http_needs_allow_http(self):
        self.setup_archive({}, lambda q: [])
        with self.assertRaisesRegex(SystemExit, "allow-http"):
            self.run_pool([{"key": "t", "name": "T", "queries": [{"target": "T"}]}], allow_http=False)
        self.assertEqual(x.prefer_https("http://bldata.berkeley.edu/a.h5"), "https://bldata.berkeley.edu/a.h5")
        self.assertEqual(x.prefer_https("http://blpd0.ssl.berkeley.edu/a.h5"), "http://blpd0.ssl.berkeley.edu/a.h5")

    def test_files_are_chosen_from_distinct_sessions_nearest_the_frequency(self):
        entries = [{"url": f"http://h/{n}.0000.h5", "file_type": "HDF5", "mjd": mjd, "center_freq": f}
                   for n, (mjd, f) in enumerate([(58602.1, 1000), (58602.2, 980), (58610.0, 990), (58620.0, 3000)])]
        entries.append({"url": "http://h/raw.0000.raw", "file_type": "baseband data", "mjd": 58630.0, "center_freq": 982})
        target = {"key": "proxcen", "freq_mhz": 982.0, "freq_window_mhz": 100}
        chosen = x.choose_files(entries, target, 7, wanted=2)
        self.assertEqual([e["url"] for e in chosen], ["http://h/1.0000.h5", "http://h/2.0000.h5"],
                         "nearest the frequency, one per session; never baseband data or a file far off in frequency")
        self.assertEqual([e["url"] for e in x.choose_files(entries, target, 7)], ["http://h/1.0000.h5", "http://h/2.0000.h5", "http://h/0.0000.h5"],
                         "then the rest, from sessions already used")


class FormatTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        x._remotes.clear()

    def test_sigproc_filterbank_files_read_like_hdf5_with_packed_angles(self):
        rng = np.random.default_rng(4)
        data = rng.normal(size=(16, 8 * 512)).astype(np.float32)
        fil = self.folder / "scan.fil"
        foff = -1 / 512  # Parkes's 1 MHz coarse channels, in 512 fine channels here
        write_fil(fil, data, telescope_id=4, source_name="PROXCEN", fch1=1000.0, foff=foff, tsamp=16.78,
                  tstart=58602.5, src_raj=142942.95, src_dej=-624046.2)
        full, header = x.load(fil)
        np.testing.assert_array_equal(full, data)
        ra, dec = x.pointing(header)
        self.assertAlmostEqual(ra, 14 + 29 / 60 + 42.95 / 3600)
        self.assertAlmostEqual(dec, -(62 + 40 / 60 + 46.2 / 3600))
        self.assertEqual(x.sky_of(header)["telescope"], "Parkes")
        self.assertEqual(x.sky_of(header)["constellation"], "Centaurus")
        self.assertEqual(x.channels_per_coarse(header), 512)
        # Parkes's coarse channels are centred on whole MHz: half a channel before the
        # fch1-aligned blocks. Channel 3 * 512 + 7 is in the coarse channel spanning 1280-1791.
        freq = 1000.0 + foff * (3 * 512 + 7)
        part, part_header = x.load(fil, freq)
        np.testing.assert_array_equal(part, data[:, 1280:1792])
        self.assertEqual(part_header["coarse_channel"], 3)
        self.assertAlmostEqual(part_header["fch1"], 1000.0 + foff * 1280)
        server = FileServer({"/scan.fil": fil.read_bytes()})
        self.addCleanup(server.close)
        with mock.patch.object(x.RangeFile, "WHOLE", 0):
            remote, _ = x.load(f"{server.base}/scan.fil", freq)
        np.testing.assert_array_equal(remote, part)

    def test_fine_channels_per_coarse_channel_follow_the_telescope_and_the_channel_width(self):
        gbt, parkes = {"telescope_id": 6}, {"telescope_id": 4}
        self.assertEqual(x.channels_per_coarse({**gbt, "foff": -2.7939677238464355e-06}), 1 << 20)
        self.assertEqual(x.channels_per_coarse({**gbt, "foff": -2.835503418452676e-06}), 1_033_216, "2016's 2.84 Hz files")
        self.assertEqual(x.channels_per_coarse({**parkes, "foff": 2.0e-06}), 500_000)
        self.assertEqual(x.channels_per_coarse({**parkes, "foff": 2.0e-06, "nfpc": 1 << 19}), 1 << 19, "the header's nfpc wins")
        self.assertEqual(x.channels_per_coarse({**gbt, "foff": -3.3e-06}), x.COARSE_WIDTH, "not a whole number: the default")
        self.assertEqual(x.coarse_shift({**parkes, "foff": 2.0e-06}, 500_000), 250_000)
        self.assertEqual(x.coarse_shift({**gbt, "foff": -2.79e-06}, 1 << 20), 0)
        spike = np.ones((16, 1024), np.float32)
        spike[:, 512] = 30
        self.assertGreater(x.dc_spike(spike), 20)
        self.assertAlmostEqual(x.dc_spike(np.ones((16, 1024), np.float32)), 1.0)

    def test_long_scans_give_their_first_16_steps_and_short_ones_none(self):
        data = np.arange(20 * 64, dtype=np.float32).reshape(20, 64)
        path = self.folder / "long.h5"
        write_h5(path, data)
        first, header = x.load(path, steps=16)
        np.testing.assert_array_equal(first, data[:16])
        self.assertEqual(header["time_steps"], 20)
        _, entry = x.make_entry(path, argparse.Namespace(product=None, source_base_url=None), header, first, "s", "stars", "hit", {})
        self.assertEqual(entry["source"]["time_steps"], "the first 16 of 20")
        with self.assertRaises(x.SkipFile):
            x.load(path, steps=24)

    def test_blind_windows_are_drawn_away_from_the_edges_and_the_dc_spike(self):
        n, width = 1 << 20, 256
        starts = x.blind_starts(n, width, 500, 7, "key", Path("scan.h5"), 0)
        self.assertEqual(starts, x.blind_starts(n, width, 500, 7, "key", Path("scan.h5"), 0), "fixed by the seed")
        self.assertNotEqual(starts, x.blind_starts(n, width, 500, 8, "key", Path("scan.h5"), 0))
        for first in starts:
            self.assertTrue(n // 20 <= first <= n - n // 20 - width)
            self.assertFalse(n // 2 - width - width // 2 < first < n // 2 + width // 2)


class SkyTests(unittest.TestCase):
    def test_constellations_come_from_the_iau_boundaries(self):
        cases = {
            "Ophiuchus": (17.211245 * 15, 12.403782),  # Voyager 1, 2020
            "Hercules": (17.211245 * 15, 13.403784),  # the off-source position 1 degree north
            "Ursa Minor": (37.9546, 89.2641),  # Polaris, around the pole
            "Octans": (317.195, -88.956),
            "Serpens Caput": (236.067, 6.4256),  # Unukalhai: Serpens is in two parts
            "Serpens Cauda": (275.327, -2.8988),
        }
        for name, (ra, dec) in cases.items():
            self.assertEqual(skycoords.constellation_at(ra, dec), name)
        with self.assertRaisesRegex(ValueError, "no constellation"):
            skycoords.constellation_at(10, 10, boundaries=[])

    def test_slices_record_where_the_telescope_pointed(self):
        on = x.sky_of(ON_2020)
        self.assertEqual((on["ra_hours"], on["dec_deg"]), (17.211245, 12.403782))
        self.assertEqual(on["epoch"], "2020-07-16T22:13:56Z")
        self.assertEqual(on["constellation"], "Ophiuchus")
        self.assertEqual(on["telescope"], "Green Bank Telescope")
        self.assertEqual(on["distance_label"], "≈150 AU from the Sun")
        reference = {"source_name": "VOYAGER-1", "sky": on}
        north = x.sky_of({**ON_2020, "src_dej": 13.403784444444447, "source_name": "VOYAGER-1 OFF"}, reference)
        east = x.sky_of({**ON_2020, "src_raj": 17.279503611111107, "src_dej": 12.403769444444446, "source_name": "VOYAGER-1 OFF"}, reference)
        self.assertEqual(north["distance_label"], "Empty sky, 1.0° north of Voyager 1")
        self.assertEqual(north["constellation"], "Hercules")
        self.assertEqual(east["distance_label"], "Empty sky, 1.0° east of Voyager 1")
        self.assertIsNone(x.sky_of({"tstart": 60000.25, "source_name": "TEST"}), "no pointing in the header, no sky block")

    def test_voyager_1_recedes_about_3_6_au_a_year(self):
        self.assertEqual(x.voyager1_distance(57650.78), "≈136 AU from the Sun")  # 2016-09-19
        self.assertEqual(x.voyager1_distance(59046.93), "≈150 AU from the Sun")  # 2020-07-16

    def test_entries_carry_the_sky_block_but_not_the_reference(self):
        args = argparse.Namespace(product=None, source_base_url=None)
        header = {**ON_2020, "src_dej": 13.403784444444447, "source_name": "VOYAGER-1 OFF",
                  "fch1": 8421.38671875, "foff": -2.7939677238464355e-06, "tsamp": 18.253611008}
        _, entry = x.make_entry(Path("scan.h5"), args, header, np.zeros((16, 4), np.float32), "s", "offsource", "carrier window",
                                {"reference": {"source_name": "VOYAGER-1", "sky": x.sky_of(ON_2020)}})
        self.assertNotIn("reference", entry["source"])
        self.assertEqual(entry["source"]["sky"]["distance_label"], "Empty sky, 1.0° north of Voyager 1")


if __name__ == "__main__":
    unittest.main()
