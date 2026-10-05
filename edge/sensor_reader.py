"""
RPi4 Hardware Sensor Reader — Production Hardened Driver (Phase 2).

Hardware Sensors Integrated:
1. ADXL345: 3-Axis Digital Accelerometer on I2C Bus 1 at address 0x53 (via smbus2).
   - Full resolution mode (+/-16g, scale factor 0.0039 g/LSB).
   - Calculates dynamic vibration RMS (subtracting 1.0g gravity).
   - Computes peak acceleration and 32-sample kurtosis (bearing/fan impulse shock detector).
2. INMP441: Digital I2S MEMS Microphone on ALSA Card 3 (plughw:3,0).
   - 44.1 kHz, 32-bit S32_LE burst capture via ALSA.
   - Calculates normalized acoustic RMS and FFT dominant peak frequency.
3. DHT22 (Optional): Digital temperature & humidity sensor on GPIO4.

Graceful Degradation:
- If hardware is missing or disconnected (e.g. running on dev machine or wire disconnected),
  it catches the failure, logs a warning once, returns last known/nominal values,
  and continues operation without crashing the daemon.
"""

import os
import sys
import time
import math
import struct
import shutil
import logging
import threading
import subprocess
from typing import Dict, Any, Optional
import numpy as np

# Configure logger
logger = logging.getLogger("sensor_reader")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ─── 1. I2C / smbus2 Driver Availability (ADXL345) ──────────────────────────
try:
    from smbus2 import SMBus
    SMBUS_AVAILABLE = True
except ImportError:
    SMBus = None
    SMBUS_AVAILABLE = False
    logger.info("smbus2 library not available -- ADXL345 will use graceful fallback")

# ─── 2. Optional DHT22 Driver Availability ──────────────────────────────────
try:
    import board as dht_board
    import adafruit_dht
    DHT22_AVAILABLE = True
except (ImportError, NotImplementedError):
    DHT22_AVAILABLE = False

# ADXL345 Registers & Constants
ADXL345_DEFAULT_ADDR = 0x53
REG_DEVID = 0x00
REG_POWER_CTL = 0x2D
REG_DATA_FORMAT = 0x31
REG_DATAX0 = 0x32
ADXL345_SCALE_FACTOR = 0.0039  # 3.9 mg/LSB in full-resolution mode
EXPECTED_DEVID = 0xE5


class SensorReader:
    """
    Unified hardware sensor reader for Raspberry Pi 4 edge node.
    Interfaces directly with ADXL345 (I2C) and INMP441 (I2S ALSA).
    """

    def __init__(self, i2c_bus: int = 1, adxl_addr: int = ADXL345_DEFAULT_ADDR,
                 alsa_device: str = "plughw:3,0", dht_pin: int = 4, max_retries: int = 3):
        self.i2c_bus_num = i2c_bus
        self.adxl_addr = adxl_addr
        self.alsa_device = os.environ.get("ALSA_MIC_DEVICE", alsa_device)
        self.dht_pin = dht_pin
        self.max_retries = max_retries

        self.bus = None
        self.adxl_available = False
        self.inmp441_available = False
        self.dht = None
        self.dht_available = False

        self._adxl_failures = 0
        self._mic_failures = 0
        self._dht_failures = 0
        self._max_consecutive_failures = 10

        # Nominal / last-known values for graceful degradation
        self._last_vibration = {
            "x": 0.0, "y": 0.0, "z": 1.0,
            "gravity_norm": 1.0, "rms": 0.02,
            "peak_accel": 0.02, "kurtosis": 3.0,
            "source": "fallback"
        }
        self._last_acoustic = {
            "rms": 0.02, "peak_freq": 0.0,
            "sample_rate": 44100, "source": "fallback"
        }
        self._last_environment = {
            "temperature": 25.0, "humidity": 50.0,
            "source": "fallback"
        }

        self._lock = threading.Lock()
        self._latest_all = {
            "vibration": dict(self._last_vibration),
            "environment": dict(self._last_environment),
            "acoustic": dict(self._last_acoustic),
            "ambient_temp": 25.0,
            "humidity": 50.0,
            "vibration_rms": 0.02,
            "peak_acceleration": 0.02,
            "vibration_kurtosis": 3.0,
            "acoustic_rms": 0.02,
            "acoustic_peak_freq": 0.0,
            "sensors_active": {"adxl345": False, "inmp441": False, "dht22": False}
        }

        self._init_adxl345()
        self._init_inmp441()
        self._init_dht22()

    def _init_adxl345(self):
        """Initialize ADXL345 accelerometer on I2C bus."""
        if not SMBUS_AVAILABLE:
            return
        try:
            self.bus = SMBus(self.i2c_bus_num)
            devid = self.bus.read_byte_data(self.adxl_addr, REG_DEVID)
            if devid == EXPECTED_DEVID:
                # Measurement mode (enable bit 3 in POWER_CTL)
                self.bus.write_byte_data(self.adxl_addr, REG_POWER_CTL, 0x08)
                # +/-16g range, full resolution (0x0B in DATA_FORMAT)
                self.bus.write_byte_data(self.adxl_addr, REG_DATA_FORMAT, 0x0B)
                time.sleep(0.02)
                self.adxl_available = True
                logger.info(f"ADXL345 accelerometer initialized (I2C Bus {self.i2c_bus_num}, Address 0x{self.adxl_addr:02X}, DEVID 0x{devid:02X})")
            else:
                logger.warning(f"ADXL345 unexpected DEVID 0x{devid:02X} (expected 0x{EXPECTED_DEVID:02X})")
        except Exception as e:
            logger.info(f"ADXL345 hardware not detected on I2C-{self.i2c_bus_num}: {e}")
            self.adxl_available = False

    def _init_inmp441(self):
        """Probe INMP441 ALSA capture card."""
        arecord_bin = shutil.which("arecord")
        if not arecord_bin:
            logger.info("arecord binary not present -- INMP441 will use fallback")
            return
        try:
            # Capture tiny 64-sample test probe to verify device responds
            cmd = ["arecord", "-D", self.alsa_device, "-c2", "-r44100", "-fS32_LE", "-s", "64", "-t", "raw"]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2.0)
            if proc.returncode == 0 and len(proc.stdout) > 0:
                self.inmp441_available = True
                logger.info(f"INMP441 MEMS microphone initialized (ALSA device '{self.alsa_device}', 44.1 kHz, 32-bit S32_LE)")
            else:
                logger.info(f"INMP441 ALSA card probe inactive ({self.alsa_device})")
        except Exception as e:
            logger.info(f"INMP441 device probe skipped: {e}")
            self.inmp441_available = False

    def _init_dht22(self):
        """Initialize optional DHT22 temperature/humidity sensor."""
        if DHT22_AVAILABLE:
            try:
                pin = getattr(dht_board, f'D{self.dht_pin}')
                self.dht = adafruit_dht.DHT22(pin, use_pulseio=False)
                self.dht_available = True
                logger.info(f"DHT22 initialized on GPIO{self.dht_pin}")
            except Exception as e:
                logger.info(f"DHT22 init skipped: {e}")
                self.dht = None

    def read_vibration(self) -> Dict[str, Any]:
        """
        Read ADXL345 accelerometer over I2C.
        Takes 32 burst samples in full resolution (+/-16g, 3.9 mg/LSB).
        Computes dynamic vibration RMS, peak acceleration, and 32-sample kurtosis.
        """
        if not self.adxl_available or self._adxl_failures >= self._max_consecutive_failures:
            return dict(self._last_vibration)

        for attempt in range(self.max_retries):
            try:
                samples_x, samples_y, samples_z, raw_mags = [], [], [], []
                for _ in range(32):
                    data = self.bus.read_i2c_block_data(self.adxl_addr, REG_DATAX0, 6)
                    x = data[0] | (data[1] << 8)
                    if x & (1 << 15):
                        x -= (1 << 16)
                    y = data[2] | (data[3] << 8)
                    if y & (1 << 15):
                        y -= (1 << 16)
                    z = data[4] | (data[5] << 8)
                    if z & (1 << 15):
                        z -= (1 << 16)

                    gx = x * ADXL345_SCALE_FACTOR
                    gy = y * ADXL345_SCALE_FACTOR
                    gz = z * ADXL345_SCALE_FACTOR
                    mag = math.sqrt(gx**2 + gy**2 + gz**2)

                    samples_x.append(gx)
                    samples_y.append(gy)
                    samples_z.append(gz)
                    raw_mags.append(mag)
                    time.sleep(0.001)

                avg_x = float(np.mean(samples_x))
                avg_y = float(np.mean(samples_y))
                avg_z = float(np.mean(samples_z))
                total_gravity = math.sqrt(avg_x**2 + avg_y**2 + avg_z**2)

                # Dynamic vibration RMS: subtract 1.0g Earth resting gravity
                vibration_rms = abs(total_gravity - 1.0)

                # Peak dynamic acceleration
                peak_accel = float(np.max(np.abs(np.array(raw_mags) - 1.0)))

                # 32-sample Kurtosis (impulse shock detection for bearing/fan fault)
                vib_series = np.array(raw_mags) - 1.0
                mean_v = np.mean(vib_series)
                var_v = np.mean((vib_series - mean_v) ** 2)
                if var_v > 1e-9:
                    kurtosis = float(np.mean((vib_series - mean_v) ** 4) / (var_v ** 2))
                else:
                    kurtosis = 3.0  # Normal Gaussian baseline

                result = {
                    "x": round(avg_x, 4),
                    "y": round(avg_y, 4),
                    "z": round(avg_z, 4),
                    "gravity_norm": round(total_gravity, 4),
                    "rms": round(vibration_rms, 4),
                    "peak_accel": round(peak_accel, 4),
                    "kurtosis": round(kurtosis, 2),
                    "source": "adxl345"
                }
                self._last_vibration = result
                self._adxl_failures = 0
                return result

            except Exception as e:
                delay = 0.05 * (2 ** attempt)
                logger.debug(f"ADXL345 read attempt {attempt+1} failed ({e}), retrying in {delay:.2f}s")
                time.sleep(delay)
                try:
                    if self.bus:
                        self.bus.write_byte_data(self.adxl_addr, REG_POWER_CTL, 0x08)
                except Exception:
                    pass

        self._adxl_failures += 1
        if self._adxl_failures == 1:
            logger.warning(f"ADXL345 read failed after {self.max_retries} retries -- using last known reading")
        return dict(self._last_vibration)

    def read_acoustic(self) -> Dict[str, Any]:
        """
        Read INMP441 I2S MEMS microphone via ALSA capture.
        Captures a 0.2s burst at 44.1 kHz, 32-bit S32_LE (8,820 samples).
        Calculates normalized dynamic acoustic RMS and dominant peak frequency using FFT.
        """
        if not self.inmp441_available or self._mic_failures >= self._max_consecutive_failures:
            return dict(self._last_acoustic)

        # 8,820 samples * 2 channels * 4 bytes/sample = 70,560 bytes
        expected_bytes = 70560
        cmd = [
            "arecord",
            "-D", self.alsa_device,
            "-c2", "-r44100", "-fS32_LE",
            "-s", "8820",
            "-t", "raw"
        ]

        for attempt in range(self.max_retries):
            try:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=1.5
                )
                if proc.returncode != 0 or len(proc.stdout) < expected_bytes:
                    raise IOError(f"arecord failed with code {proc.returncode} (bytes={len(proc.stdout)})")

                raw_bytes = proc.stdout[:expected_bytes]
                num_ints = len(raw_bytes) // 4
                samples = struct.unpack(f"<{num_ints}i", raw_bytes)

                # Left channel (L/R pin tied to GND)
                left_channel = np.array(samples[0::2], dtype=np.float64)

                # Normalized RMS: divide raw 32-bit amplitude by 2^31
                raw_rms = np.sqrt(np.mean(left_channel ** 2))
                normalized_rms = float(raw_rms / (2 ** 31))

                # Dominant peak frequency via FFT
                fft_vals = np.abs(np.fft.rfft(left_channel))
                freqs = np.fft.rfftfreq(len(left_channel), d=1.0 / 44100.0)
                if len(fft_vals) > 1:
                    # Ignore index 0 (DC component)
                    peak_idx = int(np.argmax(fft_vals[1:]) + 1)
                    peak_freq = float(freqs[peak_idx])
                else:
                    peak_freq = 0.0

                result = {
                    "rms": round(normalized_rms, 6),
                    "peak_freq": round(peak_freq, 1),
                    "sample_rate": 44100,
                    "source": "inmp441"
                }
                self._last_acoustic = result
                self._mic_failures = 0
                return result

            except Exception as e:
                delay = 0.1 * (2 ** attempt)
                logger.debug(f"INMP441 read attempt {attempt+1} failed ({e}), retrying in {delay:.2f}s")
                time.sleep(delay)

        self._mic_failures += 1
        if self._mic_failures == 1:
            logger.warning(f"INMP441 read failed after {self.max_retries} retries -- using last known reading")
        return dict(self._last_acoustic)

    def read_environment(self) -> Dict[str, Any]:
        """Read DHT22 ambient temperature and humidity with graceful fallback."""
        if not self.dht_available or self.dht is None or self._dht_failures >= self._max_consecutive_failures:
            return dict(self._last_environment)

        for attempt in range(self.max_retries):
            try:
                temp = self.dht.temperature
                hum = self.dht.humidity
                if temp is None or hum is None:
                    raise RuntimeError("DHT22 returned None")
                result = {
                    "temperature": round(temp, 1),
                    "humidity": round(hum, 1),
                    "source": "dht22"
                }
                self._last_environment = result
                self._dht_failures = 0
                return result
            except Exception as e:
                time.sleep(0.5 * (2 ** attempt))

        self._dht_failures += 1
        return dict(self._last_environment)

    def read_all(self, force: bool = True) -> Dict[str, Any]:
        """
        Read all connected hardware sensors and return unified telemetry dict.
        Thread-safe: protects I2C and ALSA devices from concurrent access.
        If force=False and lock is busy, returns cached latest reading immediately.
        """
        acquired = self._lock.acquire(blocking=force, timeout=2.0 if force else 0.0)
        if not acquired:
            return dict(self._latest_all)

        try:
            vibration = self.read_vibration()
            environment = self.read_environment()
            acoustic = self.read_acoustic()

            res = {
                "vibration": vibration,
                "environment": environment,
                "acoustic": acoustic,
                "ambient_temp": environment.get("temperature", 25.0),
                "humidity": environment.get("humidity", 50.0),
                "vibration_rms": vibration.get("rms", 0.02),
                "peak_acceleration": vibration.get("peak_accel", 0.02),
                "vibration_kurtosis": vibration.get("kurtosis", 3.0),
                "acoustic_rms": acoustic.get("rms", 0.02),
                "acoustic_peak_freq": acoustic.get("peak_freq", 0.0),
                "sensors_active": {
                    "adxl345": vibration.get("source") == "adxl345",
                    "inmp441": acoustic.get("source") == "inmp441",
                    "dht22": environment.get("source") == "dht22"
                }
            }
            self._latest_all = res
            return res
        finally:
            self._lock.release()

    def get_latest(self) -> Dict[str, Any]:
        """Non-blocking access to the latest cached physical sensor reading."""
        return dict(self._latest_all)

    def get_status(self) -> Dict[str, str]:
        """Return diagnostic status for all hardware sensors."""
        return {
            "adxl345": "OK" if self.adxl_available and self._adxl_failures < self._max_consecutive_failures else ("DISABLED" if self._adxl_failures >= self._max_consecutive_failures else "UNAVAILABLE"),
            "inmp441": "OK" if self.inmp441_available and self._mic_failures < self._max_consecutive_failures else ("DISABLED" if self._mic_failures >= self._max_consecutive_failures else "UNAVAILABLE"),
            "dht22": "OK" if self.dht_available and self._dht_failures < self._max_consecutive_failures else ("DISABLED" if self._dht_failures >= self._max_consecutive_failures else "UNAVAILABLE")
        }


if __name__ == "__main__":
    print("=" * 65)
    print("  RPi4 Hardware Sensor Reader (ADXL345 + INMP441) Diagnostic")
    print("=" * 65)
    reader = SensorReader()
    print(f"Sensor Initialization Status: {reader.get_status()}")

    print("\nStreaming 3 live sensor readings (interval 1.0s)...")
    for i in range(3):
        data = reader.read_all()
        print(f"\n[Sample {i+1}]")
        vib = data["vibration"]
        mic = data["acoustic"]
        print(f"  Vibration (ADXL345): RMS={vib.get('rms', 0.0):.4f}g | Norm={vib.get('gravity_norm', 1.0):.3f}g | Peak={vib.get('peak_accel', 0.0):.3f}g | Kurtosis={vib.get('kurtosis', 3.0):.2f} [{vib.get('source')}]")
        print(f"  Acoustic  (INMP441): RMS={mic.get('rms', 0.0):.6f} | PeakFreq={mic.get('peak_freq', 0.0):.1f} Hz [{mic.get('source')}]")
        print(f"  Active Flags       : {data['sensors_active']}")
        time.sleep(1.0)
    print("\nSelf-test complete.")
