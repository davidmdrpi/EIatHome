"""tools/extract_slices.py: remote range reads, coarse-channel selection and the drift search."""

import argparse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
from pathlib import Path
import re
import tempfile
import threading
import unittest
from unittest import mock

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

                # The tool reads https:// URLs only; the test server is plain HTTP on localhost.
                with mock.patch.object(x, "COARSE_WIDTH", width), mock.patch.object(x, "is_url", lambda p: isinstance(p, str)):
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
        remote = x.RangeFile(self.serve(path, honour_range=False))
        with self.assertRaisesRegex(OSError, "doesn't support range requests"):
            remote.read(16)

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
