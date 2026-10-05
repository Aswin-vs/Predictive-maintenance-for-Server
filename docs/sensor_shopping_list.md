# 🛒 External Sensor Shopping List for RPi 4 Edge Node

> **Project:** Edge-AI Predictive Maintenance for Server Hardware  
> **Purpose:** Physical sensors attached to/near the monitored server, connected to RPi 4 GPIO  
> **Date:** 2026-09-04  
> **Budget Estimate:** ₹1,500 – ₹3,000 total (all sensors)

---

## ⚠️ Important Notes Before Buying

1. **Buy from a reputable Indian electronics store** (Robu.in, ThinkRobotics, Amazon.in, or local electronics market)
2. **All sensors connect to the RPi 4** — NOT to the Patient server
3. **The RPi 4 sits physically near/on the server chassis** to measure its environment
4. **No soldering required** for any of these — all use standard jumper wires
5. **Code is already written** in [`edge/sensor_reader.py`](file:///c:/Users/LOQ/Documents/Antigravity/Predictive%20maintenance%20for%20Server/edge/sensor_reader.py) — just plug in and go

---

## 📦 Required Sensors (Core — Must Buy)

### 1. DHT22 (AM2302) — Ambient Temperature & Humidity Sensor

| Spec | Value |
|---|---|
| **What it measures** | Room temperature (°C) + Relative Humidity (%) |
| **Why it matters** | Servers in hot/humid rooms degrade faster. High humidity → condensation → short circuits. Low humidity → ESD risk. |
| **Range** | -40°C to +80°C, 0–100% RH |
| **Accuracy** | ±0.5°C, ±2% RH |
| **Interface** | Single GPIO pin (digital, 1-wire protocol) |
| **Voltage** | 3.3V or 5V (RPi 4 compatible) |
| **RPi 4 Wiring** | VCC → Pin 1 (3.3V), DATA → GPIO4 (Pin 7), GND → Pin 6 |
| **Pull-up Resistor** | 10KΩ between VCC and DATA (some modules have it built-in) |
| **Price (India)** | ₹150 – ₹250 |
| **Preferred Model** | DHT22 on breakout board (3 pins: VCC, DATA, GND — resistor built-in) |

> **⚠️ DO NOT buy DHT11** — it has only ±2°C accuracy and 0–50°C range. The DHT22 is far superior for server room monitoring.

**Where to place:** Mount on or near the server chassis air intake. This measures the ambient conditions the server is operating in.

---

### 2. ADXL345 — 3-Axis Digital Accelerometer (Vibration Sensor)

| Spec | Value |
|---|---|
| **What it measures** | Vibration in X, Y, Z axes (measured in g-force) |
| **Why it matters** | Detects failing fan bearings, HDD mechanical degradation, rack vibrations, and chassis resonance |
| **Range** | ±2g, ±4g, ±8g, ±16g (configurable) |
| **Resolution** | 13-bit (up to 4mg/LSB at ±2g) |
| **Interface** | I2C (address 0x53 default) or SPI |
| **Voltage** | 3.3V (RPi 4 compatible) |
| **Sample Rate** | Up to 3200 Hz (we use ~100 Hz for vibration RMS) |
| **RPi 4 Wiring (I2C)** | VCC → Pin 1 (3.3V), GND → Pin 6, SDA → GPIO2 (Pin 3), SCL → GPIO3 (Pin 5) |
| **Price (India)** | ₹150 – ₹300 |
| **Preferred Model** | GY-291 ADXL345 breakout module |

**Where to place:** Attach directly to the server chassis using double-sided tape or a small bracket. Place near the HDD bay or on top of the case. The sensor detects physical vibrations that indicate mechanical degradation.

**What the AI does with it:**
- Computes **Vibration RMS** (Root Mean Square of X²+Y²+Z²) — the single number that represents total vibration energy
- A rising vibration RMS trend over days/weeks indicates bearing wear in fans or HDDs
- Sudden spikes indicate mechanical shocks (someone bumped the rack, earthquake, etc.)

---

### 3. MAX4466 — Electret Microphone Amplifier (Acoustic Sensor)

| Spec | Value |
|---|---|
| **What it measures** | Sound/noise level (analog voltage proportional to acoustic energy) |
| **Why it matters** | Detects coil whine (failing capacitors), grinding fan bearings, HDD click-of-death, and unusual server room noise |
| **Frequency Range** | 20 Hz – 20 kHz |
| **Output** | Analog voltage (0–3.3V centered at VCC/2) |
| **Gain** | Adjustable via onboard potentiometer (25x – 125x) |
| **Voltage** | 2.4V – 5.5V (RPi 4 compatible at 3.3V) |
| **Interface** | Analog output → **requires MCP3008 ADC** (RPi 4 has no analog input) |
| **Price (India)** | ₹100 – ₹200 |

**Where to place:** Mount near the server's fan exhaust or near the HDD cage. Pointed at the server, not at the room. This listens for mechanical sounds that precede failure.

> **⚠️ IMPORTANT:** The RPi 4 has NO analog GPIO pins. You MUST buy the MCP3008 ADC (below) to read this sensor.

---

### 4. MCP3008 — 8-Channel 10-bit ADC (Analog-to-Digital Converter)

| Spec | Value |
|---|---|
| **What it does** | Converts analog signals (like MAX4466 output) to digital values the RPi 4 can read |
| **Why it's needed** | RPi 4 GPIO is digital-only. This chip bridges analog sensors to the Pi |
| **Channels** | 8 analog inputs (CH0–CH7) — we use CH0 for MAX4466, others available for future sensors |
| **Resolution** | 10-bit (0–1023 values) |
| **Interface** | SPI (4 wires) |
| **Voltage** | 2.7V – 5.5V |
| **RPi 4 Wiring** | VDD → 3.3V, VREF → 3.3V, AGND → GND, DGND → GND, CLK → GPIO11 (SCLK), DOUT → GPIO9 (MISO), DIN → GPIO10 (MOSI), CS → GPIO8 (CE0) |
| **Price (India)** | ₹80 – ₹150 |
| **Preferred Model** | MCP3008-I/P DIP-16 package (breadboard-friendly) |

---

## 🔌 Required Accessories

| Item | Purpose | Qty | Price (₹) |
|---|---|---|---|
| **Breadboard (400-point)** | Wiring hub for all sensors | 1 | ₹50–80 |
| **Jumper Wires (M-F, 20cm)** | Connect sensors to RPi 4 GPIO | 20-pack | ₹50–80 |
| **Jumper Wires (M-M, 20cm)** | Breadboard connections | 10-pack | ₹30–50 |
| **10KΩ Resistor** | Pull-up for DHT22 (if not built-in) | 2 | ₹5 |
| **Double-sided mounting tape** | Attach ADXL345 to server chassis | 1 roll | ₹50 |

---

## 📋 Complete Bill of Materials

| # | Component | Model | Interface | Price (₹) |
|---|---|---|---|---|
| 1 | Temperature + Humidity | **DHT22 (AM2302)** breakout | GPIO4 (digital) | 150–250 |
| 2 | 3-Axis Accelerometer | **ADXL345 (GY-291)** breakout | I2C (SDA/SCL) | 150–300 |
| 3 | Electret Microphone | **MAX4466** amplifier module | Analog → MCP3008 CH0 | 100–200 |
| 4 | ADC Converter | **MCP3008** DIP-16 | SPI (SCLK/MISO/MOSI/CE0) | 80–150 |
| 5 | Breadboard | 400-point | — | 50–80 |
| 6 | Jumper wires M-F | 20cm, 20-pack | — | 50–80 |
| 7 | Jumper wires M-M | 20cm, 10-pack | — | 30–50 |
| 8 | 10KΩ Resistors | Through-hole | — | 5 |
| | | | **TOTAL** | **₹615 – ₹1,115** |

---

## 🔧 RPi 4 GPIO Pin Mapping (All Sensors)

```
RPi 4 GPIO Header (40 pins)
┌──────────────────────────────────┐
│  Pin 1 (3.3V) ←── VCC for all   │
│  Pin 3 (GPIO2/SDA) ←── ADXL345  │
│  Pin 5 (GPIO3/SCL) ←── ADXL345  │
│  Pin 6 (GND) ←── GND for all    │
│  Pin 7 (GPIO4) ←── DHT22 DATA   │
│  Pin 19 (GPIO10/MOSI) ←── MCP3008 DIN  │
│  Pin 21 (GPIO9/MISO) ←── MCP3008 DOUT │
│  Pin 23 (GPIO11/SCLK) ←── MCP3008 CLK  │
│  Pin 24 (GPIO8/CE0) ←── MCP3008 CS     │
└──────────────────────────────────┘
```

---

## 📦 Optional / Future Sensors (Phase G+)

These are NOT needed for the current project but could enhance the system for company testing:

| Sensor | Measures | Interface | Price (₹) | Use Case |
|---|---|---|---|---|
| **BME280** | Temp + Humidity + Barometric Pressure | I2C | ₹200–350 | More accurate than DHT22, adds pressure (altitude/weather) |
| **INA219** | Current/Power draw (DC) | I2C | ₹150–250 | Measure server's actual power consumption from PSU rail |
| **GP2Y1010AU0F** | Dust/Particulate density | Analog (MCP3008) | ₹200–350 | Detect dust buildup inside server chassis |
| **MPU6050** | 6-axis Accel + Gyroscope | I2C | ₹100–200 | More axes than ADXL345, adds rotation detection |
| **DS18B20** | Precision temperature (±0.5°C) | 1-Wire | ₹80–150 | Attach directly to heatsink/chassis for contact temp |

---

## 🐍 Software Already Implemented

The sensor reading code is already written and tested in [`edge/sensor_reader.py`](file:///c:/Users/LOQ/Documents/Antigravity/Predictive%20maintenance%20for%20Server/edge/sensor_reader.py):

```python
# Required Python packages (already in requirements_edge.txt):
# pip install adafruit-circuitpython-adxl34x  # ADXL345
# pip install adafruit-circuitpython-dht       # DHT22
# pip install adafruit-circuitpython-mcp3xxx   # MCP3008
# pip install adafruit-blinka                  # CircuitPython on RPi4
```

**Graceful Degradation:** If any sensor is missing or fails, the code automatically falls back to simulated values (configurable defaults). The system works with ALL sensors, SOME sensors, or NO sensors.

---

## 🛠️ Installation Steps (After Purchasing)

1. **Enable I2C and SPI on RPi 4:**
   ```bash
   sudo raspi-config
   # Navigate: Interface Options → I2C → Enable
   # Navigate: Interface Options → SPI → Enable
   sudo reboot
   ```

2. **Wire sensors on breadboard** (follow GPIO pin mapping above)

3. **Test each sensor individually:**
   ```bash
   # Test I2C (ADXL345 should show at address 0x53):
   sudo i2cdetect -y 1
   
   # Test DHT22:
   python3 -c "import board; import adafruit_dht; d = adafruit_dht.DHT22(board.D4); print(d.temperature, d.humidity)"
   
   # Test MCP3008 (MAX4466):
   python3 -c "import board; import busio; import digitalio; import adafruit_mcp3xxx.mcp3008 as MCP; from adafruit_mcp3xxx.analog_in import AnalogIn; spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO); cs = digitalio.DigitalInOut(board.D8); mcp = MCP.MCP3008(spi, cs); chan = AnalogIn(mcp, MCP.P0); print(f'Value: {chan.value}, Voltage: {chan.voltage}')"
   ```

4. **Restart edge-ai service:**
   ```bash
   sudo systemctl restart edge-ai
   ```
   The `sensor_reader.py` will auto-detect the newly connected sensors and start feeding real environmental data into the AI pipeline.
