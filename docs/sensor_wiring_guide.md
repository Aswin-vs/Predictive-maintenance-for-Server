# 🔌 Complete Hardware Sensor Wiring & Pinout Guide (Raspberry Pi 4)

> **Document Status:** Authoritative Hardware Reference  
> **Target Node:** Raspberry Pi 4 Model B (4GB RAM) — "The Doctor (Edge)"  
> **Purpose:** Comprehensive wiring schematic, physical pinout maps, jumper connections, and OS driver verification commands.  
> **Use Case:** Reconnecting external sensors after using the Raspberry Pi 4 for other projects or when swapping boot pendrives/SD cards.

---

## 📑 Table of Contents
1. [Executive Hardware Inventory](#1-executive-hardware-inventory)
2. [Raspberry Pi 4 40-Pin GPIO Header Reference](#2-raspberry-pi-4-40-pin-gpio-header-reference)
3. [Sensor-by-Sensor Wiring Schematics](#3-sensor-by-sensor-wiring-schematics)
   - [Sensor 1: ADXL345 3-Axis Digital Accelerometer (I2C)](#sensor-1-adxl345-3-axis-digital-accelerometer-i2c)
   - [Sensor 2: INMP441 Digital MEMS Microphone (I2S ALSA)](#sensor-2-inmp441-digital-mems-microphone-i2s-alsa)
   - [Sensor 3: DHT22 Digital Temperature & Humidity Sensor (1-Wire)](#sensor-3-dht22-digital-temperature--humidity-sensor-1-wire)
   - [Sensor 4 (Optional ADC): MCP3008 + MAX4466 Analog Microphone](#sensor-4-optional-adc-mcp3008--max4466-analog-microphone)
4. [Master Consolidated Pin Connection Table](#4-master-consolidated-pin-connection-table)
5. [Operating System Configuration (`config.txt`)](#5-operating-system-configuration-configtxt)
6. [Hardware Verification & CLI Smoke Tests](#6-hardware-verification--cli-smoke-tests)
7. [Physical Placement & Mounting Guidelines](#7-physical-placement--mounting-guidelines)

---

## 1. Executive Hardware Inventory

| Sensor / Module | Primary Function | Protocol | Physical Bus / Device | Operating Voltage | Verification Source Code |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ADXL345** (GY-291) | 3-axis chassis vibration & bearing shock (RMS, Kurtosis) | **I2C** | Bus 1, Address `0x53` | 3.3V DC | `edge/sensor_reader.py:57-80` |
| **INMP441** | High-fidelity acoustic cavitation & fan whistling FFT | **I2S** | ALSA device `plughw:3,0` | 3.3V DC | `edge/sensor_reader.py:99-130` |
| **DHT22** (AM2302) | Ambient server intake air temp (°C) & humidity (%) | **1-Wire** | BCM GPIO 4 (Pin 7) | 3.3V or 5V DC | `edge/sensor_reader.py:50-56` |
| **MCP3008 + MAX4466** *(Optional fallback)* | 8-channel 10-bit SPI ADC for analog acoustic mics | **SPI** | SPI0, CE0 (BCM GPIO 8) | 3.3V DC | `docs/sensor_shopping_list.md:85-100` |

---

## 2. Raspberry Pi 4 40-Pin GPIO Header Reference

Below is the complete physical orientation of the 40-pin header on the Raspberry Pi 4 Model B (viewed with the board oriented with USB ports facing right and GPIO header along the top edge):

```
                                  TOP EDGE OF BOARD
                         Physical Pin 1 ──┐   ┌── Physical Pin 2
                                          ▼   ▼
                           3.3V Power  [ 1] [ 2]  5V Power
             (I2C1 SDA)  GPIO 2 (SDA)  [ 3] [ 4]  5V Power
             (I2C1 SCL)  GPIO 3 (SCL)  [ 5] [ 6]  Ground (GND)
             (DHT22 DATA)      GPIO 4  [ 7] [ 8]  GPIO 14 (UART TXD)
                               Ground  [ 9] [10]  GPIO 15 (UART RXD)
                              GPIO 17  [11] [12]  GPIO 18 (I2S BCLK / PCM_CLK)
                              GPIO 27  [13] [14]  Ground (GND)
                              GPIO 22  [15] [16]  GPIO 23
                           3.3V Power  [17] [18]  GPIO 24
                    (SPI0 MOSI) GPIO 10 [19] [20]  Ground (GND)
                    (SPI0 MISO) GPIO 9  [21] [22]  GPIO 25
                    (SPI0 SCLK) GPIO 11 [23] [24]  GPIO 8 (SPI0 CE0)
                               Ground  [25] [26]  GPIO 7 (SPI0 CE1)
                              ID_SD    [27] [28]  ID_SC
                              GPIO 5   [29] [30]  Ground (GND)
                              GPIO 6   [31] [32]  GPIO 12
                              GPIO 13  [33] [34]  Ground (GND)
             (I2S WS / PCM_FS) GPIO 19  [35] [36]  GPIO 16
                              GPIO 26  [37] [38]  GPIO 20 (I2S DIN / PCM_DIN)
                               Ground  [39] [40]  GPIO 21
                                          ▲   ▲
                        Physical Pin 39 ──┘   └── Physical Pin 40
```

---

## 3. Sensor-by-Sensor Wiring Schematics

### Sensor 1: ADXL345 3-Axis Digital Accelerometer (I2C)

The ADXL345 measures mechanical chassis vibration. It calculates dynamic vibration RMS by subtracting the 1.0g gravity vector and computes 32-sample kurtosis to detect bearing dry-out or impact shocks.

```
       ADXL345 (GY-291 Breakout)                 Raspberry Pi 4 GPIO
     ┌───────────────────────────┐             ┌────────────────────────┐
     │  VCC                      │────────────▶│ Pin 1  (3.3V Power)    │
     │  GND                      │────────────▶│ Pin 6  (Ground)        │
     │  CS  (Chip Select)        │────────────▶│ Pin 17 (3.3V Power)    │ (Forces I2C mode)
     │  SDO (Alt Address / MISO) │────────────▶│ Pin 9  (Ground)        │ (Sets I2C address 0x53)
     │  SDA (I2C Data)           │────────────▶│ Pin 3  (GPIO 2 / SDA)  │
     │  SCL (I2C Clock)          │────────────▶│ Pin 5  (GPIO 3 / SCL)  │
     │  INT1 (Not Connected)     │             │                        │
     │  INT2 (Not Connected)     │             │                        │
     └───────────────────────────┘             └────────────────────────┘
```

#### Detailed Pin Connections for ADXL345:
| ADXL345 Pin | RPi4 Physical Pin | RPi4 BCM Function | Wire Color Recommendation | Technical Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **VCC** | **Pin 1** | 3.3V Power | **Red** | Primary power supply (3.3V DC, &lt; 0.1 mA). |
| **GND** | **Pin 6** | Ground | **Black** | Common ground reference. |
| **CS** | **Pin 17** (or Pin 1) | 3.3V Power | **Orange** | **CRITICAL:** Pulled HIGH to 3.3V to enable I2C bus mode (instead of SPI). |
| **SDO** | **Pin 9** (or Pin 6) | Ground | **Brown** | **CRITICAL:** Pulled LOW to Ground to set I2C address to `0x53` (default in code). |
| **SDA** | **Pin 3** | GPIO 2 (I2C1 SDA) | **Yellow** | Serial Data Line with internal RPi hardware pull-up (1.8kΩ). |
| **SCL** | **Pin 5** | GPIO 3 (I2C1 SCL) | **Green** | Serial Clock Line with internal RPi hardware pull-up (1.8kΩ). |

> **⚠️ CRITICAL ADXL345 HARDWARE NOTE:**  
> If the `CS` pin is left floating, the sensor will intermittently freeze or switch to SPI mode. Always tie `CS` directly to 3.3V.  
> If `SDO` is tied to 3.3V instead of Ground, the I2C address shifts to `0x54`, causing `sensor_reader.py` to raise a device missing exception. Keep `SDO` grounded.

---

### Sensor 2: INMP441 Digital MEMS Microphone (I2S ALSA)

The INMP441 is an omnidirectional MEMS microphone that outputs 24-bit/32-bit digital audio directly over the I2S bus. It provides continuous acoustic telemetry at 44.1 kHz, detecting fan motor cavitation and coil whine without an external analog ADC.

```
       INMP441 MEMS Microphone                   Raspberry Pi 4 GPIO
     ┌───────────────────────────┐             ┌────────────────────────┐
     │  VDD                      │────────────▶│ Pin 1  (3.3V Power)    │
     │  GND                      │────────────▶│ Pin 14 (Ground)        │
     │  L/R (Channel Select)     │────────────▶│ Pin 20 (Ground)        │ (Sets Left Channel)
     │  WS  (Word Select/LRCLK)  │────────────▶│ Pin 35 (GPIO 19 / PCM_FS)
     │  SCK (Bit Clock / BCLK)   │────────────▶│ Pin 12 (GPIO 18 / PCM_CLK)
     │  SD  (Serial Data / DIN)  │────────────▶│ Pin 38 (GPIO 20 / PCM_DIN)
     └───────────────────────────┘             └────────────────────────┘
```

#### Detailed Pin Connections for INMP441:
| INMP441 Pin | RPi4 Physical Pin | RPi4 BCM Function | Wire Color Recommendation | Technical Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **VDD** | **Pin 1** (or 17) | 3.3V Power | **Red** | Power supply (3.3V DC, clean rail). |
| **GND** | **Pin 14** | Ground | **Black** | Common ground reference. |
| **L/R** | **Pin 20** | Ground | **Brown** | Pulled LOW to Ground to configure Left audio channel output. |
| **WS** | **Pin 35** | GPIO 19 (PCM_FS) | **Blue** | Word Select / Frame Sync (44.1 kHz clock). |
| **SCK** | **Pin 12** | GPIO 18 (PCM_CLK) | **Purple** | Continuous serial bit clock (~2.82 MHz). |
| **SD** | **Pin 38** | GPIO 20 (PCM_DIN) | **White** | Serial data stream entering RPi PCM input. |

> **⚠️ CRITICAL INMP441 HARDWARE NOTE:**  
> The INMP441 is a 3.3V device only. Connecting VDD to 5V will instantly destroy the MEMS transducer.  
> The RPi4 hardware PCM interface is bound to GPIO 18, 19, and 20. These pins are fixed in hardware and cannot be remapped to other GPIOs.

---

### Sensor 3: DHT22 Digital Temperature & Humidity Sensor (1-Wire)

The DHT22 measures the ambient room temperature and relative humidity at the server intake grill.

```
       DHT22 / AM2302 Sensor                     Raspberry Pi 4 GPIO
     ┌───────────────────────────┐             ┌────────────────────────┐
     │  Pin 1: VCC               │────────────▶│ Pin 2  (5V) or Pin 1 (3.3V)
     │  Pin 2: DATA              │────────────▶│ Pin 7  (GPIO 4)        │
     │  Pin 3: NC (No Connect)   │             │                        │
     │  Pin 4: GND               │────────────▶│ Pin 6  (Ground)        │
     └───────────────────────────┘             └────────────────────────┘
            ▲             ▲
            └───[ 10KΩ ]──┘  (Pull-up resistor between VCC and DATA)
            (Note: Breakout modules with 3 pins already include this resistor!)
```

#### Detailed Pin Connections for DHT22:
| DHT22 Pin | RPi4 Physical Pin | RPi4 BCM Function | Wire Color Recommendation | Technical Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **VCC** | **Pin 1** (3.3V) or **Pin 2** (5V) | Power | **Red** | 3.3V to 5V DC (3.3V recommended for direct GPIO safety). |
| **DATA** | **Pin 7** | GPIO 4 | **Yellow** | Bidirectional 1-wire digital timing pulse line. |
| **GND** | **Pin 6** (or 9) | Ground | **Black** | Ground return. |

> **💡 BREAKOUT BOARD TIP:**  
> If you have the 3-pin PCB version (labeled `+`, `out`, `-`), the 10kΩ pull-up resistor is already soldered on board. Wire directly: `+` to Pin 1, `out` to Pin 7, `-` to Pin 6.

---

### Sensor 4 (Optional ADC): MCP3008 + MAX4466 Analog Microphone

If using an analog microphone (such as MAX4466) instead of the digital INMP441, the MCP3008 8-channel SPI ADC is used:

```
       MCP3008 ADC (DIP-16)                      Raspberry Pi 4 GPIO
     ┌───────────────────────────┐             ┌────────────────────────┐
     │ Pin 16: VDD (Power)       │────────────▶│ Pin 1  (3.3V Power)    │
     │ Pin 15: VREF (Reference)  │────────────▶│ Pin 1  (3.3V Power)    │
     │ Pin 14: AGND (Analog GND) │────────────▶│ Pin 6  (Ground)        │
     │ Pin 13: CLK (SPI Clock)   │────────────▶│ Pin 23 (GPIO 11 / SCLK)│
     │ Pin 12: DOUT (MISO)       │────────────▶│ Pin 21 (GPIO 9 / MISO) │
     │ Pin 11: DIN (MOSI)        │────────────▶│ Pin 19 (GPIO 10 / MOSI)│
     │ Pin 10: CS (Chip Select)  │────────────▶│ Pin 24 (GPIO 8 / CE0)  │
     │ Pin 9:  DGND (Digital GND)│────────────▶│ Pin 25 (Ground)        │
     │                           │             │                        │
     │ Pin 1:  CH0 (Analog Input)│◀────────────│ MAX4466 OUT pin        │
     └───────────────────────────┘             └────────────────────────┘
```

---

## 4. Master Consolidated Pin Connection Table

When rewiring all sensors onto a single 40-pin header, use the following consolidated mapping to prevent pin sharing conflicts:

| Physical Pin # | Header Pin Name | Connected To | Wire Color | Notes |
| :---: | :--- | :--- | :---: | :--- |
| **1** | **3.3V Power** | ADXL345 `VCC`, INMP441 `VDD`, DHT22 `+` | **Red** | Use breadboard or multi-jumper splice |
| **2** | 5V Power | *(Unused / Reserved)* | — | Reserved for fans / 5V accessories |
| **3** | **GPIO 2 (I2C1 SDA)** | **ADXL345 `SDA`** | **Yellow** | Hardware I2C Data |
| **4** | 5V Power | *(Unused)* | — | — |
| **5** | **GPIO 3 (I2C1 SCL)** | **ADXL345 `SCL`** | **Green** | Hardware I2C Clock |
| **6** | **Ground (GND)** | **ADXL345 `GND`**, DHT22 `GND` | **Black** | Common Ground |
| **7** | **GPIO 4** | **DHT22 `DATA`** | **Yellow** | 1-Wire Digital Signal |
| **8** | GPIO 14 (TXD) | *(Unused)* | — | UART Console |
| **9** | **Ground (GND)** | **ADXL345 `SDO`** | **Brown** | Sets ADXL345 I2C address to `0x53` |
| **10** | GPIO 15 (RXD) | *(Unused)* | — | UART Console |
| **11** | GPIO 17 | *(Unused)* | — | General Purpose |
| **12** | **GPIO 18 (PCM_CLK)** | **INMP441 `SCK`** | **Purple** | I2S Bit Clock Line |
| **13** | GPIO 27 | *(Unused)* | — | General Purpose |
| **14** | **Ground (GND)** | **INMP441 `GND`** | **Black** | Microphone Ground |
| **15** | GPIO 22 | *(Unused)* | — | General Purpose |
| **16** | GPIO 23 | *(Unused)* | — | General Purpose |
| **17** | **3.3V Power** | **ADXL345 `CS`** | **Orange** | Forces ADXL345 into I2C Mode |
| **18** | GPIO 24 | *(Unused)* | — | General Purpose |
| **19** | GPIO 10 (MOSI) | *(MCP3008 `DIN` if used)* | White | Optional SPI ADC |
| **20** | **Ground (GND)** | **INMP441 `L/R`** | **Brown** | Selects Left Channel Audio |
| **21** | GPIO 9 (MISO) | *(MCP3008 `DOUT` if used)* | Grey | Optional SPI ADC |
| **23** | GPIO 11 (SCLK) | *(MCP3008 `CLK` if used)* | Purple | Optional SPI ADC |
| **24** | GPIO 8 (CE0) | *(MCP3008 `CS` if used)* | Blue | Optional SPI ADC |
| **35** | **GPIO 19 (PCM_FS)** | **INMP441 `WS`** | **Blue** | I2S Word Select / LRCLK Line |
| **38** | **GPIO 20 (PCM_DIN)** | **INMP441 `SD`** | **White** | I2S Serial Data Line |
| **39** | Ground (GND) | Shield Ground | Black | Optional cable shielding |

---

## 5. Operating System Configuration (`config.txt`)

When setting up a fresh boot drive (USB pendrive or MicroSD card) with Raspberry Pi OS Lite (64-bit), the I2C and I2S hardware overlays must be enabled in `/boot/firmware/config.txt` (or `/boot/config.txt` on older OS releases):

```bash
sudo nano /boot/firmware/config.txt
```

Append the following hardware overlay configuration blocks at the end of the file:

```ini
# ==============================================================================
# Edge-AI Predictive Maintenance Sensor Bus Overlays
# ==============================================================================

# Enable Hardware I2C Bus 1 at 400kHz fast mode (for ADXL345)
dtparam=i2c_arm=on
dtparam=i2c_arm_baudrate=400000

# Enable Hardware SPI Bus (if using MCP3008 ADC)
dtparam=spi=on

# Enable Hardware I2S Digital Audio Interface (for INMP441 Microphone)
dtparam=i2s=on
dtoverlay=googlevoicehat-soundcard

# Disable onboard HDMI audio to prevent ALSA card index swapping
dtoverlay=vc4-kms-v3d,noaudio
```

Save and reboot the Raspberry Pi 4:
```bash
sudo reboot
```

---

## 6. Hardware Verification & CLI Smoke Tests

After reconnecting the wires and booting up, execute the following commands in the RPi4 terminal to verify that each sensor is recognized:

### Test 1: Verify I2C Bus and ADXL345 Address
Run `i2cdetect` to scan I2C Bus 1:
```bash
sudo apt-get install -y i2c-tools
i2cdetect -y 1
```
**Expected Output:**
You should see address `53` clearly populated in the matrix:
```
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00:                         -- -- -- -- -- -- -- -- 
10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
30: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
40: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
50: -- -- -- 53 -- -- -- -- -- -- -- -- -- -- -- -- 
60: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
70: -- -- -- -- -- -- -- --                         
```

### Test 2: Quick Python Read of ADXL345 Vibration
Run a one-line Python test to query the Device ID register (`0x00`):
```bash
python3 -c "import smbus2; b=smbus2.SMBus(1); devid=b.read_byte_data(0x53, 0x00); print(f'ADXL345 Device ID: {hex(devid)} (Expected: 0xe5)')"
```
**Expected Output:**
```
ADXL345 Device ID: 0xe5 (Expected: 0xe5)
```

### Test 3: Verify ALSA Soundcard for INMP441
Check that ALSA detects the I2S microphone hardware card:
```bash
arecord -l
```
**Expected Output:**
You should see the voicehat or I2S recording device listed (typically `card 3, device 0`):
```
**** List of CAPTURE Hardware Devices ****
card 3: sndrpigooglevoi [snd_rpi_googlevoicehat_soundcar], device 0: Google voiceHAT SoundCard HiFi voicehat-hifi-0 [Google voiceHAT SoundCard HiFi voicehat-hifi-0]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
```

### Test 4: Record a Test Audio Burst via ALSA
Record a 2-second WAV burst at 44.1 kHz, 32-bit S32_LE:
```bash
arecord -D plughw:3,0 -c 1 -r 44100 -f S32_LE -t wav -d 2 test_mic.wav
ls -lh test_mic.wav
```
If `test_mic.wav` is created with a size of ~352 KB, the INMP441 digital I2S microphone is operating at 100% fidelity.

### Test 5: End-to-End Driver Test via `sensor_reader.py`
Run the built-in standalone diagnostic suite in `edge/sensor_reader.py`:
```bash
cd /home/pi/predictive-maintenance
python3 edge/sensor_reader.py
```
**Expected Output:**
```
[INFO] Initialized ADXL345 on I2C bus 1 (addr 0x53)
[INFO] ALSA capture device available: plughw:3,0
=== SensorReader Production Hardware Test ===
Iteration 1:
  Vibration: RMS=0.0219g (Peak=0.0381g, Kurtosis=3.12)
  Acoustics: RMS=0.1084 (Peak Freq=120.0 Hz)
  Ambient:   Temp=26.4°C, Humidity=54.2%
[OK] All physical hardware sensors verified functional!
```

---

## 7. Physical Placement & Mounting Guidelines

1. **ADXL345 Accelerometer:**
   - **Mounting Method:** Use thin, firm 3M double-sided foam tape or a rigid 3D-printed clip.
   - **Chassis Location:** Attach directly to the metal chassis wall immediately adjacent to the primary drive cage (or top cover above the CPU heatsink). Do NOT mount onto plastic fan shrouds, as flexible plastic absorbs high-frequency bearing vibrations.
   - **Orientation:** Z-axis perpendicular to the server base plate. The driver automatically subtracts the steady-state 1.0g gravity vector along the vertical axis.
2. **INMP441 Digital Microphone:**
   - **Mounting Method:** Mount with the small sound port hole facing inward toward the primary cooling fan exhaust grill or HDD spindle motor.
   - **Distance:** Keep within 5–15 cm of the fan assembly to capture bearing micro-chatter and cavitation noise while minimizing room echo.
3. **DHT22 Ambient Sensor:**
   - **Mounting Method:** Position in free-floating ambient air near the server chassis front air intake. Keep away from warm exhaust vents or power supply heat sinks to ensure accurate room temperature measurements.
